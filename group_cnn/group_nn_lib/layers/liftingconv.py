import torch
import math


class LiftingKernelBase(torch.nn.Module):
    def __init__(self, group, kernel_size, in_channels, out_channels):
        """ Implements a base class for the lifting kernel. Stores R^2 grid
        over which the lifting kernel is defined and it's transformed copies
        under the action of a group H

        """
        super().__init__()
        self.group = group

        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Creating spatial kernel grid. There are the coordinates on which our
        # kernel weights are defined
        self.register_buffer("grid_R2", torch.stack(torch.meshgrid(
            torch.linspace(-1. ,1., self.kernel_size),
            torch.linspace(-1. ,1., self.kernel_size),
            indexing = 'ij'
        )).to(self.group.identity.device))

        # Transform the grid by the elements in this group
        self.register_buffer("transformed_grid_R2", self.create_transformed_grid_R2())

    def create_transformed_grid_R2(self):
        """ Transform the created grid by the group action of each group element.
        This yields a grid (over H) of spatial grids (over R2). In other words,
        a list of grids, each index of which is the original spatial grid transformed by
        a corresponding group element of H

        """

        group_elements = self.group.elements()

        transformed_grid = torch.stack([self.group.left_action_on_R2(el, self.grid_R2) for el in self.group.inverse(group_elements)], dim = 1)

        return transformed_grid

    def sample(self, sampled_group_elements):
        """ Sample convolution kernels for a given number of group elements


        arguments should include:
        @param sampled_group_elements: the group elements over which to sample
            the convolution kernels

        should return:
        @return kernels: filter band extending over all input channels,
            containing kernels transformed for all output group elements.
        """

        raise NotImplementedError()
        
        
class InterpolativeLiftingKernel(LiftingKernelBase):

    def __init__(self, group, kernel_size, in_channels, out_channels):
        super().__init__(group, kernel_size, in_channels, out_channels)

        # Create and initialize a set of weights, we will interpolate these
        # to create our transformed spatial kernels
        self.weight = torch.nn.Parameter(torch.zeros((
            self.out_channels,
            self.in_channels,
            self.kernel_size,
            self.kernel_size,
        ), device = self.group.identity.device))

        # Initialize weights using kaiming uniform inialization
        torch.nn.init.kaiming_uniform_(self.weight.data, a = math.sqrt(5))

    def sample(self):
        """ Sample convolution kernels for a given number of group elements


        should return:
        @return kernels: filter bank extending over all input channels,
            containing kernels transformed for all output group elements
        """
        # First, we fold the output channel dim into input channel dim.
        weight = self.weight.view(
            self.out_channels*self.in_channels,
            self.kernel_size,
            self.kernel_size
        )

        transformed_weight = []
        for spatial_grid_idx in range(self.group.elements().numel()):
            transformed_weight.append(
                bilinear_interpolation(weight, self.transformed_grid_R2[:, spatial_grid_idx, :, :])
            )

        transformed_weight = torch.stack(transformed_weight)

        # Separate input and output channels.
        transformed_weight = transformed_weight.view(
            self.group.elements().numel(),
            self.out_channels,
            self.in_channels,
            self.kernel_size,
            self.kernel_size
        )

        # Put out channel dimension  before group dimension. We do this
        # to be able to use pytorched Conv2D. Details below!
        transformed_weight = transformed_weight.transpose(0, 1)

        return transformed_weight
    
    
    
class LiftingConvolution(torch.nn.Module):

    def __init__(self, group, in_channels, out_channels, kernel_size, padding):
        super().__init__()

        self.kernel = InterpolativeLiftingKernel(
            group = group,
            kernel_size = kernel_size,
            in_channels = in_channels,
            out_channels = out_channels
        )

        self.padding = padding

    def forward(self, x):
        """ Perform lifting convolution

        @ param x: Input sample [batch_dim, in_channels, spatial_dim_1,
                spatial_dim_2]

        @ return: Function on a homogeneous space of the group
            [batch_dim, out_channels, num_group_elements, spatial_dim_1,
            spatial_dim_2]
        """

        conv_kernels = self.kernel.sample().reshape(
            self.kernel.out_channels*self.kernel.group.elements().numel(),
            self.kernel.in_channels,
            self.kernel.kernel_size,
            self.kernel.kernel_size
        )

        conv_kernels = conv_kernels

        x = torch.nn.functional.conv2d(
            input = x,
            weight = conv_kernels,
            padding = self.padding)

        x = x.view(
            -1,
            self.kernel.out_channels,
            self.kernel.group.elements().numel(),
            x.shape[-1],
            x.shape[-2]
        )


        return x