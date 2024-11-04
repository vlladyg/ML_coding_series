import logging
from typing import Optional, List

import torch
import torch.nn.functional
from torch_runstats.scatter import scatter

from nequip.data import AtomicDataDict
from nequip.data.transforms import TypeMapper
from nequip.utils import dtype_from_name
from nequip.utils.versions import _TORCH_IS_GE_1_13
from .rescale import RescaleOutput


from .gmm import GraphModuleMixin

class PerSpeciesScaleShift(GraphModuleMixin, torch.nn.Module):
    """Scale and/or shift a predicted per-atom property based on (learnable) per-species/type parameters.

    Note that scaling/shifting is always done (casting into) ``default_dtype``, even if ``model_dtype`` is lower precision.

    Args:
        field: the per-atom field to scale/shift.
        num_types: the number of types in the model.
        shifts: the initial shifts to use, one per atom type.
        scales: the initial scales to use, one per atom type.
        arguments_in_dataset_units: if ``True``, says that the provided shifts/scales are in dataset
            units (in which case they will be rescaled appropriately by any global rescaling later
            applied to the model); if ``False``, the provided shifts/scales will be used without modification.

            For example, if identity shifts/scales of zeros and ones are provided, this should be ``False``.
            But if scales/shifts computed from the training data are used, and are thus in dataset units,
            this should be ``True``.
        out_field: the output field; defaults to ``field``.
    """

    field: str
    out_field: str
    scales_trainble: bool
    shifts_trainable: bool
    has_scales: bool
    has_shifts: bool
    default_dtype: torch.dtype
    _use_fma: bool

    def __init__(
        self,
        field: str,
        num_types: int,
        type_names: List[str],
        shifts: Optional[List[float]],
        scales: Optional[List[float]],
        arguments_in_dataset_units: bool,
        out_field: Optional[str] = None,
        scales_trainable: bool = False,
        shifts_trainable: bool = False,
        default_dtype: Optional[str] = None,
        irreps_in={},
    ):
        super().__init__()
        self.num_types = num_types
        self.type_names = type_names
        self.field = field
        self.out_field = f"shifted_{field}" if out_field is None else out_field
        self._init_irreps(
            irreps_in=irreps_in,
            my_irreps_in={self.field: "0e"},  # input to shift must be a single scalar
            irreps_out={self.out_field: irreps_in[self.field]},
        )

        self.default_dtype = dtype_from_name(
            torch.get_default_dtype() if default_dtype is None else default_dtype
        )

        self.has_shifts = shifts is not None
        if shifts is not None:
            shifts = torch.as_tensor(shifts, dtype=self.default_dtype)
            if len(shifts.reshape([-1])) == 1:
                shifts = (
                    torch.ones(num_types, dtype=shifts.dtype, device=shifts.device)
                    * shifts
                )
            assert shifts.shape == (num_types,), f"Invalid shape of shifts {shifts}"
            self.shifts_trainable = shifts_trainable
            if shifts_trainable:
                self.shifts = torch.nn.Parameter(shifts)
            else:
                self.register_buffer("shifts", shifts)
        else:
            self.register_buffer("shifts", torch.Tensor())

        self.has_scales = scales is not None
        if scales is not None:
            scales = torch.as_tensor(scales, dtype=self.default_dtype)
            if len(scales.reshape([-1])) == 1:
                scales = (
                    torch.ones(num_types, dtype=scales.dtype, device=scales.device)
                    * scales
                )
            assert scales.shape == (num_types,), f"Invalid shape of scales {scales}"
            self.scales_trainable = scales_trainable
            if scales_trainable:
                self.scales = torch.nn.Parameter(scales)
            else:
                self.register_buffer("scales", scales)
        else:
            self.register_buffer("scales", torch.Tensor())

        assert isinstance(arguments_in_dataset_units, bool)
        self.arguments_in_dataset_units = arguments_in_dataset_units

        # we can use FMA for performance but its type promotion is broken until 1.13
        self._use_fma = _TORCH_IS_GE_1_13

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:

        if not (self.has_scales or self.has_shifts):
            return data

        species_idx = data[AtomicDataDict.ATOM_TYPE_KEY].squeeze(-1)
        in_field = data[self.field]
        assert len(in_field) == len(
            species_idx
        ), "in_field doesnt seem to have correct per-atom shape"

        if self._use_fma and self.has_scales and self.has_shifts:
            # we can used an FMA for performance
            # addcmul computes
            # input + tensor1 * tensor2 elementwise
            # it will promote to widest dtype, which comes from shifts/scales
            in_field = torch.addcmul(
                torch.index_select(self.shifts, 0, species_idx).view(-1, 1),
                torch.index_select(self.scales, 0, species_idx).view(-1, 1),
                in_field,
            )
        else:
            # fallback path for torch<1.13 OR mix of enabled shifts and scales
            # multiplication / addition promotes dtypes already, so no cast is needed
            # this is specifically because self.*[species_idx].view(-1, 1)
            # is never a scalar (ndim == 0), since it is always [n_atom, 1]
            if self.has_scales:
                in_field = (
                    torch.index_select(self.scales, 0, species_idx).view(-1, 1)
                    * in_field
                )
            if self.has_shifts:
                in_field = (
                    torch.index_select(self.shifts, 0, species_idx).view(-1, 1)
                    + in_field
                )
        data[self.out_field] = in_field
        return data

    def update_for_rescale(self, rescale_module: RescaleOutput):
        if not self.arguments_in_dataset_units:
            # nothing to rescale, arguments are in normalized units already / unitless
            return
        # are we scaling something related to the global rescaling?
        if self.field not in rescale_module.scale_keys:
            return
        # now check that we have the right rescaling in the specific energy case
        if self.field == AtomicDataDict.PER_ATOM_ENERGY_KEY and not (
            set(rescale_module.scale_keys) <= set(AtomicDataDict.ALL_ENERGY_KEYS)
        ):
            raise AssertionError("Some unsupported energy scaling arangement...")
        if self.arguments_in_dataset_units and rescale_module.has_scale:
            logging.debug(
                f"PerSpeciesScaleShift's arguments were in dataset units; rescaling:\n  "
                f"Original scales: {TypeMapper.format(self.scales, self.type_names) if self.has_scales else 'n/a'} "
                f"shifts: {TypeMapper.format(self.shifts, self.type_names) if self.has_shifts else 'n/a'}"
            )
            with torch.no_grad():
                if self.has_scales:
                    self.scales.div_(rescale_module.scale_by)
                if self.has_shifts:
                    self.shifts.div_(rescale_module.scale_by)
            logging.debug(
                f"  New scales: {TypeMapper.format(self.scales, self.type_names) if self.has_scales else 'n/a'} "
                f"shifts: {TypeMapper.format(self.shifts, self.type_names) if self.has_shifts else 'n/a'}"
            )
