import torch
import math

class GroupKernelBase(torch.nn.Module):

    def __init__(self, group, kernel_size, in_channels, out_channels):
        """ Implements base class for the group convolution kernel.stores grid
        defined over the group R^2 \rtimes H and it's transformed copies under
        all elements of the group H

        """
        super().__init__()
        self.group = group

        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Create a spatial kernel grid
        self.register_buffer("grid_R2", torch.stack(torch.meshgrid(
            torch.linspace(-1., 1., self.kernel_size),
            torch.linspace(-1., 1., self.kernel_size),
            indexing = 'ij'
        )).to(self.group.identity.device))

        # The kernel grid now also extends over the group H, as our input
        # feature maps contain an additional group dimension
        self.register_buffer("grid_H", self.group.elements())
        self.register_buffer("transformed_grid_R2xH", self.create_transformed_grid_R2xH())


    def create_transformed_grid_R2xH(self):
        """ Transform the created grid over R^2 \rtimes H by the group action of
        each group element in H

        This yields a set of grids over the group. In other words, a list of
        grids, each index of which is the original grid over G transformed by
        a corresponding group element in H.
        """

        # Sample the group H.
        group_elements = self.grid_H


        # Transform the grid defined over R2 with the sampled group elements.
        # We again would like to end up with a grid of shape [2, |H|, kernel_size, kernel_size].
        transformed_grip_R2 = torch.stack([self.group.left_action_on_R2(el, self.grid_R2) for el in self.group.inverse(group_elements)], dim = 1)

        # Transform the grid defined over H with the sampled group elements. We want a grid of
        # shape [|H|, |H|]. Make sure to stack the transformed like above (over the 1st dim).
        transformed_grid_H = torch.stack([self.group.product(el, group_elements) for el in self.group.inverse(group_elements)], dim = 1)

        # Rescale values to between -1 and 1, we do this to please the torch
        # grid_sample function.
        transformed_grid_H = self.group.normalize_group_elements(transformed_grid_H)

        # Create a combined grid as the product of the grids over R2 and H
        # repeat R2 along the group dimension, and repeat H along the spatial dimension
        # to create a [3, |H|, |H|, kernel_size, kernel_size] grid
        transformed_grid = torch.cat(
            (
                transformed_grip_R2.view(
                    2,
                    group_elements.numel(),
                    1,
                    self.kernel_size,
                    self.kernel_size
                ).repeat(1, 1, group_elements.numel(), 1, 1),
                transformed_grid_H.view(
                    1,
                    group_elements.numel(),
                    group_elements.numel(),
                    1,
                    1,
                ).repeat(1, 1, 1, self.kernel_size, self.kernel_size)
            ),
            dim = 0
        )



        return transformed_grid

    def sample(self, sampled_group_elements):
        """ Sample convolution kernels for a given number of group elements

        arguments should include:
        @param sampled_group_elements: the group elements over which to sample
            the convolution kernels

        should return:
        @ return kernels: filter bank extending over all input channels,
            containing kernels transformed for all output group elements.
        """

        raise NotImplementedError()
        
        
class InterpolativeGroupKernel(GroupKernelBase):

    def __init__(self, group, kernel_size, in_channels, out_channels):
        super().__init__(group, kernel_size, in_channels, out_channels)

        # create and initialise a set of weights, we will interpolate these
        # to create our transformed spatial kernels. Note that our weight
        # now also extends over the group H.
        self.weight = torch.nn.Parameter(torch.zeros((
            self.out_channels,
            self.in_channels,
            self.group.elements().numel(),
            self.kernel_size,
            self.kernel_size,
        ), device = self.group.identity.device))


        # initialize weights using kaiming uniform initialization
        torch.nn.init.kaiming_uniform_(self.weight.data, a = math.sqrt(5))

    def sample(self):
        """ Sample convolution kernels for a given number of group elements

        should return:
        @ return kernels: filter bank extending over all input channels,
            containing kernels transformed for all output group elements.
        """

        # First, we fold the output channel dim into the input channel dim;
        # this allows us to transform the entire filter bank in one go using the
        # interpolation function.
        weight = self.weight.view(
            self.out_channels * self.in_channels,
            self.group.elements().numel(),
            self.kernel_size,
            self.kernel_size
        )


        # We loop over all group elements and retrieve weight values for
        # the corresponding transformed grids over R2xH.
        transformed_weight = torch.stack([trilinear_interpolation(weight, self.transformed_grid_R2xH[:, grid_idx, :, :, :])
                                          for grid_idx in range(self.group.elements().numel())])

        transformed_weight = transformed_weight.view(
            self.group.elements().numel(),
            self.out_channels,
            self.in_channels,
            self.group.elements().numel(),
            self.kernel_size,
            self.kernel_size
        )

        # Put out channel dimension before group dimension. We do this
        # to be able to use pytorched Conv2D. Details below!
        transformed_weight = transformed_weight.transpose(0, 1)

        return transformed_weight
    
    
class GroupConvolution(torch.nn.Module):


    def __init__(self, group, in_channels, out_channels, kernel_size, padding):
        super().__init__()

        self.kernel = InterpolativeGroupKernel(
            group = group,
            kernel_size = kernel_size,
            in_channels = in_channels,
            out_channels = out_channels
        )

        self.padding = padding


    def forward(self, x):
        """ Perform group convolution

        @param x: Input sample [batch_dim, in_channels, group_dim, spatial_dim_1,
            spatial_dim_2]

        @ return: Function on a homogeneous space of the group
            [batch_dim, out_channels, num_group_elements, spatial_dim_1,
             spatial_dim_2]
        """

        # We now fold the group dimensions of our input into the input channel
        # dimension.
        x = x.reshape(-1,
            x.shape[1] * x.shape[2],
            x.shape[3],
            x.shape[4]
        )

        # We obtain convolution kernels transformed under the group.
        conv_kernels = self.kernel.sample()

        conv_kernels = conv_kernels.reshape(
            self.kernel.out_channels * self.kernel.group.elements().numel(),
            self.kernel.in_channels * self.kernel.group.elements().numel(),
            self.kernel.kernel_size,
            self.kernel.kernel_size,
        )

        # Apply group convolution, note that the reshape folds the 'output' group
        # dimension of the kernel into the output channel dimension, and the
        # 'input' group dimension into the input channel dimension.

        x = torch.nn.functional.conv2d(input = x,
                                       weight = conv_kernels,
                                       padding = self.padding)

        # Reshape [batch_dim, in_channels * num_group_elements, spatial_dim_1,
        # spatial_dim_2] into [batch_dim, in_channels, num_group_elements,
        # spatial_dim_1, spatial_dim_2], separating channel and group
        # dimensions.
        x = x.view(
            -1,
            self.kernel.out_channels,
            self.kernel.group.elements().numel(),
            x.shape[-1],
            x.shape[-2],
        )

        return x