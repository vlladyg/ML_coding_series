def bilinear_interpolation(signal, grid):
    """ Obtain signal values for a set of gridpoints through bilinear interpolation

    @param signal: Tensor containing pixel values [C, H, W] or [N, C, H, W]
    @param grid: Tensor containing coordinate values [2, Hf, Wf] or [2, N, Hf, Wf]

    @return signal_out: Tensor containing pixel values [C, Hf, Wf] or [N, C, Hf, Wf]
    """
    # If signal or grid is a 3D array, add a dimension to support grid_sample.
    # This means adding batch dimension
    if len(signal.shape) == 3:
        signal = signal.unsqueeze(0)
    if len(grid.shape) == 3:
        grid = grid.unsqueeze(1)

    # Grid sample expects [N, H, W, 2] instead of [2, N, H, W]
    grid = grid.permute(1, 2, 3, 0)

    # Grid sample expects YX instead of XY
    grid = torch.roll(grid, shifts = 1, dims = -1)

    signal_out = torch.nn.functional.grid_sample(signal,
                                                 grid,
                                                 padding_mode='zeros',
                                                 align_corners = True,
                                                 mode = 'bilinear',
                                                )


    return signal_out

def trilinear_interpolation(signal, grid):
    """ Obtain signal values for a set of gridpoints through trilinear interpolation

    @param signal: Tensor containing pixel values [C, D, H, W] or [N, C, D, H, W]
    @param grid: Tensor containing coordinate values [2, Df, Hf, Wf] or [2, N, Df, Hf, Wf]

    @return signal_out: Tensor containing pixel values [C, Df, Hf, Wf] or [N, C, Df, Hf, Wf]
    """
    # If signal or grid is a 3D array, add a dimension to support grid_sample.
    # This means adding batch dimension
    if len(signal.shape) == 4:
        signal = signal.unsqueeze(0)
    if len(grid.shape) == 4:
        grid = grid.unsqueeze(1)

    # Grid sample expects [N, H, W, 2] instead of [2, N, H, W]
    grid = grid.permute(1, 2, 3, 4, 0)

    # Grid sample expects YX instead of XY
    grid = torch.roll(grid, shifts = 1, dims = -1)

    signal_out = torch.nn.functional.grid_sample(signal,
                                                 grid,
                                                 padding_mode='zeros',
                                                 align_corners = True,
                                                 mode = 'bilinear', # actually trilinear in this case ...
                                                )


    return signal_out