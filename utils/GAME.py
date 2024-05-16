import numpy as np
import torch


def padding(array, values, axis):
    # This function should be doing post padding 0s.
    if axis not in {0, 1}:
        print("Error! axis should be 0 or 1.")

    dim = array.shape
    new_dim = [0, 0]
    for i in range(2):
        if i == axis:
            new_dim[i] = dim[i] + 1
        else:
            new_dim[i] = dim[i]
    new_dim = tuple(new_dim)
    new_array = torch.zeros(new_dim)

    for i in range(dim[0]):
        for j in range(dim[1]):
            new_array[i][j] = array[i][j]
    return new_array


def adjust_dim(array):
    # Make the dim even
    if array.shape[0] % 2 != 0:
        array = padding(array, 0, 0)
    if array.shape[1] % 2 != 0:
        array = padding(array, 0, 1)
    return array


def GAME_recursive(density, gt, currentLevel, targetLevel):
    if currentLevel == targetLevel:
        game = abs(torch.sum(density) - torch.sum(gt))
        return game

    else:
        y_half = int(density.shape[0] / 2)
        x_half = int(density.shape[1] / 2)

        density_slice = [];
        gt_slice = []

        density_slice.append(density[0:y_half, 0:x_half])
        density_slice.append(density[0:y_half, x_half:density.shape[1]])
        density_slice.append(density[y_half:density.shape[0], 0:x_half])
        density_slice.append(density[y_half:density.shape[0], x_half:density.shape[1]])

        gt_slice.append(gt[0:y_half, 0:x_half])
        gt_slice.append(gt[0:y_half, x_half:density.shape[1]])
        gt_slice.append(gt[y_half:density.shape[0], 0:x_half])
        gt_slice.append(gt[y_half:density.shape[0], x_half:density.shape[1]])


        currentLevel = currentLevel + 1
        res = []
        for a in range(4):
            res.append(GAME_recursive(density_slice[a], gt_slice[a], currentLevel, targetLevel))
        game = sum(res)
        return torch.round(game)


def GAME_metric(preds, gts, l):
    res = []
    for i in range(gts.shape[0]):
        res.append(GAME_recursive(preds[i][0], gts[i], 0, l))
    return torch.mean(torch.tensor(res))