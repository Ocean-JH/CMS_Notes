#!/usr/bin/env python
# -*- coding: utf-8 -*-
from ase import io
import numpy as np


def lindemann(filename, n_start=None, n_end=None):
    """
    Calculate the Lindemann index for atomic vibrations using the provided trajectory.

    Parameters:
    - filename: Name of the file containing atomic positions (typically a LAMMPS dump file).
    - n_start: Starting index for the trajectory.
    - n_end: Ending index for the trajectory.

    Returns:
    - Lindemann index value.
    """

    # Construct the slice string based on provided indices
    if n_start is None and n_end is None:
        slice_str = ":"
    elif n_start is None:
        slice_str = f":{n_end}"
    elif n_end is None:
        slice_str = f"{n_start}:"
    else:
        slice_str = f"{n_start}:{n_end}"

    # Read atomic positions from the file for the given range
    atoms = io.read(filename, slice_str, 'lammps-dump-text')

    # Total number of atoms and total number of timesteps
    N = atoms[0].positions.shape[0]
    N_t = len(atoms)
    print('number of atoms in each frame =', N, 'number of frames =', N_t)

    # Initialize arrays to store distances between atoms
    r_square = np.zeros((N_t, N, N))

    # Loop through each timestep
    for t in range(N_t):
        G = np.dot(atoms[t].positions, atoms[t].positions.T)
        H = np.tile(np.diag(G), (N, 1))

        # Compute the squared distance between each pair of atoms at time t
        r_square[t, :, :] = H + H.T - 2 * G

    # Compute average distance and squared average distance over all timesteps
    r_ave = np.mean(np.sqrt(r_square), axis=0)
    print(r_square.shape)
    print(r_ave.shape)
    print('r_ave_max =', np.max(r_ave), 'r_ave_min =', np.min(r_ave))

    r2_ave = np.mean(r_square, axis=0)
    print('r2_ave =', np.max(r2_ave))

    # Extract upper triangular part of matrices
    r_ave_upper = r_ave[np.triu_indices(N, k=1)]
    print('r_ave_upper_max =', np.max(r_ave_upper), 'r_ave_upper_min =', np.min(r_ave_upper))
    r2_ave_upper = r2_ave[np.triu_indices(N, k=1)]

    # Calculate Lindemann criterion for upper triangular elements
    value_upper = np.sqrt(r2_ave_upper - np.power(r_ave_upper, 2)) / r_ave_upper

    # Sum over all unique atom pairs
    total_value = np.sum(value_upper)

    # Normalize by the total number of atom pairs
    return total_value * 2 / (N * (N - 1))


if __name__ == '__main__':
    for i in range(900, 1110, 10):
        filename = r'fusion_{}.atom'.format(i)
        lindemann_index = lindemann(filename)
        print(i, lindemann_index)
