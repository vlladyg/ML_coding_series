import torch
from torch.nn import AdaptiveAvgPool3d
from layers.liftingconv import LiftingConvolution
from layers.groupconv import GroupConvolution


class GroupEquivariantCNN(torch.nn.Module):

    def __init__(self, group, in_channels, out_channels, kernel_size, num_hidden, hidden_channels):
        super().__init__()

        # Create the lifting convolution
        self.lifting_conv = LiftingConvolution(group = group,
                                               in_channels = in_channels,
                                               out_channels = hidden_channels,
                                               kernel_size = kernel_size,
                                               padding=0
                                              )

        # Create a set of group convolutions.
        self.gconvs = torch.nn.ModuleList()

        for i in range(num_hidden):
            self.gconvs.append(
                GroupConvolution(
                    group = group,
                    in_channels = hidden_channels,
                    out_channels = hidden_channels,
                    kernel_size = kernel_size,
                    padding = 0
                )
            )

        # Create the projection layer. Hint: check the import at the top of
        # this cell.
        self.projection_layer = torch.nn.AdaptiveAvgPool3d(1)

        # And a final linear layer for classification
        self.final_linear = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, x):
        # Lift and disentangle features in the input.
        x = self.lifting_conv(x)
        x = torch.nn.functional.layer_norm(x, x.shape[-4:])
        x = torch.nn.functional.relu(x)

        # Apply group convolutions.
        for gconv in self.gconvs:
            x = gconv(x)
            x = torch.nn.functional.layer_norm(x, x.shape[-4:])
            x = torch.nn.functional.relu(x)

        # to ensure equivariance, apply max pooling over group and spatial dims
        x = self.projection_layer(x).squeeze()

        x = self.final_linear(x)
        return x



class CNN(torch.nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, num_hidden, hidden_channels):
        super().__init__()

        self.first_conv = torch.nn.Conv2d(
            in_channels = in_channels,
            out_channels = hidden_channels,
            kernel_size = kernel_size,
            padding = 0
        )

        self.convs = torch.nn.ModuleList()
        for i in range(num_hidden):
            self.convs.append(
                torch.nn.Conv2d(
                    in_channels = hidden_channels,
                    out_channels = hidden_channels,
                    kernel_size = kernel_size,
                    padding = 0
                )
            )

        self.final_linear = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, x):
        x = self.first_conv(x)
        x = torch.nn.functional.layer_norm(x, x.shape[-3:])
        x = torch.nn.functional.relu(x)

        for conv in self.convs:
            x = conv(x)
            x = torch.nn.functional.layer_norm(x, x.shape[-3:])
            x = torch.nn.functional.relu(x)

        # Apply average pooling over the ramaining spatial dimensions.
        x = torch.nn.functional.adaptive_avg_pool2d(x, 1).squeeze()

        x = self.final_linear(x)
        return x