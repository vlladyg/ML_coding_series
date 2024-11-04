import logging
from typing import Optional, List

import torch
import torch.nn.functional
from torch_runstats.scatter import scatter

from e3nn.o3 import Linear

from nequip.data import AtomicDataDict
from nequip.data.transforms import TypeMapper
from nequip.utils import dtype_from_name
from nequip.utils.versions import _TORCH_IS_GE_1_13

from .gmm import GraphModuleMixin

class AtomwiseReduce(GraphModuleMixin, torch.nn.Module):
    constant: float

    def __init__(
        self,
        field: str,
        out_field: Optional[str] = None,
        reduce="sum",
        avg_num_atoms=None,
        irreps_in={},
    ):
        super().__init__()
        assert reduce in ("sum", "mean", "normalized_sum")
        self.constant = 1.0
        if reduce == "normalized_sum":
            assert avg_num_atoms is not None
            self.constant = float(avg_num_atoms) ** -0.5
            reduce = "sum"
        self.reduce = reduce
        self.field = field
        self.out_field = f"{reduce}_{field}" if out_field is None else out_field
        self._init_irreps(
            irreps_in=irreps_in,
            irreps_out=(
                {self.out_field: irreps_in[self.field]}
                if self.field in irreps_in
                else {}
            ),
        )

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        field = data[self.field]
        if AtomicDataDict.BATCH_KEY in data:
            result = scatter(
                field,
                data[AtomicDataDict.BATCH_KEY],
                dim=0,
                dim_size=len(data[AtomicDataDict.BATCH_PTR_KEY]) - 1,
                reduce=self.reduce,
            )
        else:
            # We can significantly simplify and avoid scatters
            if self.reduce == "sum":
                result = field.sum(dim=0, keepdim=True)
            elif self.reduce == "mean":
                result = field.mean(dim=0, keepdim=True)
            else:
                assert False
        if self.constant != 1.0:
            result = result * self.constant
        data[self.out_field] = result
        return data