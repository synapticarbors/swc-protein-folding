import argparse
import time

import numpy as np
import numba as nb


def read_coordinates(xyzfile):
    '''Read a .xyz formatted file and extract coordinates.
    This ignores atom types. Returns an (n_atoms, 3)
    numpy array'''

    # This could be replaced by reading the file and parsing it
    # line-by-line. 

    # The first two rows are header info, and then we are going
    # to ignore the first column since it contains the atom type
    coords = np.loadtxt(xyzfile, skiprows=2, usecols=(1,2,3))

    return coords

def write_coordinates(xyzfile, coords):
    '''Write the coords to `xyzfile`, appending if the file already exists'''
    n_atoms = coords.shape[0]

    with open(xyzfile, 'a') as f:
        f.write('{:d}\n'.format(n_atoms))
        f.write('\n')  # Comment line -- possible write frame number

        for i in range(n_atoms):
            f.write('C {:4.3f} {:4.3f} {:4.3f}\n'.format(*coords[i]))


@nb.jit(nopython=True)
def calculate_energy(d, d0):
    '''Calculate the energy of the system given the pairwise distance between
    atoms, stored in the (n_atoms, n_atoms) 2D array `d` and the folded reference
    distances stored in an array of the same shape, `d0`.
    
    E = E_bonded + E_nat + E_nb
    '''

    E = (calculate_bonded_energy(d, d0) 
            +  calculate_nativecontact_energy(d, d0) 
            + calculate_nonbonded_energy(d, d0))

    return E


@nb.jit(nopython=True)
def calculate_bonded_energy(d, d0):
    '''Calculate the bond energy between consecutive pairs of atoms
    along the chain. The energy is that of a harmonic spring with rest
    length equal to the distance in the native structure'''

    assert d.shape == d0.shape
    n_atoms = d.shape[0]
    E = 0.0

    for i in range(n_atoms - 1):
        E += (d[i,i+1] - d0[i,i+1])**2

    return E


@nb.jit(nopython=True)
def calculate_nativecontact_energy(d, d0):
    '''Calculate the Go-like energy term that stabilizes native contacts'''

    n_atoms = d.shape[0]
    E = 0.0
    cutoff = 7.0 # units of angstroms

    # loop over all pairs of atoms, (i,j) subject to i < j
    # if d0[i,j] < cutoff then atoms are "in contact" in the native 
    # structure
    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):

            if d0[i,j] <= cutoff:
                sigma = 2.0**(-1.0/6.0) * d0[i,j]
                x = sigma / d[i,j]
                E += 4.0*(x**12 - x**6)

    return E


@nb.jit(nopython=True)
def calculate_nonbonded_energy(d, d0):
    '''Calculate the repulsive interaction between pairs of atoms
    that do not form native contacts'''

    n_atoms = d.shape[0]
    E = 0.0
    native_cutoff = 7.0  # This value is repeated in the nativecontacts terms and should be parameterized
    nb_cutoff = 3.0
    sigma = 2.0**(-1.0/6.0) * nb_cutoff

    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):

            if d0[i,j] > native_cutoff:
                x = sigma / d[i,j]
                if d[i,j] < nb_cutoff:
                    E += 4.0*(x**12 - x**6)

    return E


@nb.jit(nopython=True)
def calculate_distances(coords):
    '''Calculate the pairwise distances between all atoms returning an
    (n_atoms, n_atoms) array, where d_{i,j} is the Euclidean distance
    between atom i and atom j.'''

    n_atoms, n_dims = coords.shape
    d = np.zeros((n_atoms, n_atoms))

    # We will loop over all pairs, but an obvious place for speeding this up
    # is recognizing that d_{i,j} == d_{j,i} and d_{i,i] == 0.0
    for i in range(n_atoms):
        for j in range(n_atoms):
            d_ij = 0.0
            for k in range(n_dims):
                r = coords[i,k] - coords[j,k]
                d_ij += r * r

            d[i,j] = np.sqrt(d_ij)

    return d


@nb.jit(nopython=True)
def generate_trial_move(coords, delta):
    '''Displace all of the atoms in the system by a random
    amount drawn from a normal distribution with std delta
    
    Possible optimization: pass in an pre-allocated array
    for `trial_coords` so we don't keep re-allocating memory.
    
    Could also probably do this with a simple array addition'''

    # make a copy of the coordinates so we don't alter the original
    trial_coords = coords.copy()
    n_atoms, n_dims = coords.shape

    for i in range(n_atoms):
        for j in range(n_dims):
            trial_coords[i,j] += np.random.randn() * delta

    return trial_coords


def create_extended_chain(n_atoms, d):
    '''Generate the coordinates of a fully extended linear chain
    with atoms initially spaced `d` angstroms apart'''

    coords = np.zeros((n_atoms, 3))

    for i in range(1, n_atoms):
        coords[i, 0] = d * i

    return coords


@nb.jit(nopython=True)
def propagate(coords, d0, delta, n_steps):
    '''Advance the system forward by taking nsteps using a Metropolis Monte Carlo
    acceptance/rejection scheme'''

    # First get the current energy of the system
    d = calculate_distances(coords)
    E_curr = calculate_energy(d, d0)

    n_accepted = 0

    for i in range(n_steps):
        trial_coords = generate_trial_move(coords, delta)
        d_trial = calculate_distances(trial_coords)
        E_trial = calculate_energy(d_trial, d0)

        dE = E_trial - E_curr
        if dE <= 0:
            accept = True
        else:
            rn = np.random.random()
            a = np.exp(-dE)
            if rn <= a:
                accept = True
            else:
                accept = False

        if accept:
            n_accepted += 1
            E_curr = E_trial
            coords[:] = trial_coords
            d[:] = d_trial

    return coords, n_accepted


def simulate(coords, d0, n_steps, delta, save_freq, outfilename, verbose=False):
    '''simulate a system with initial coordinates `coords` and a native
    state pairwise distance matrix, `d0` for `n_steps`. Write the trajectory
    to the file `outfilename`, appending if the file already exists. Save the 
    coordinates every `save_freq` steps. Note, `n_steps / save_freq` should be
    an integer value.
    '''

    n_blocks = int(n_steps / save_freq)

    for i in xrange(n_blocks):
        b_start = time.time()

        coords, n_accepted = propagate(coords, d0, delta, save_freq)
        write_coordinates(outfilename, coords)

        b_end = time.time()

        if verbose:
            pct_complete = 100.0*(i + 1) / n_blocks
            pct_accepted = 100.0*n_accepted / save_freq
            time_elapsed = b_end - b_start
            print 'Completed {:5.1f} %; Last block took {:4.3f} s with acceptance rate: {:5.1f}\r'.format(pct_complete, time_elapsed, pct_accepted)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--n_steps', type=int, required=True, help='number of steps')
    parser.add_argument('--save_freq', type=int, required=True, help='Save coordinates every save_freq steps')
    parser.add_argument('--ref_coords', required=True, help='File containing reference coordinates')
    parser.add_argument('-o', '--outfile', default='traj.xyz', help='File name to write trajectory to')
    parser.add_argument('-d', '--delta', type=float, required=True, help='Trial move size in units of angstroms')
    parser.add_argument('--verbose', action='store_true', help='Print timing information')

    args = parser.parse_args()

    ref_coords = read_coordinates(args.ref_coords)
    d0 = calculate_distances(ref_coords)
    n_atoms = d0.shape[0]

    coords = create_extended_chain(n_atoms, 3.0)

    simulate(coords, d0, args.n_steps, args.delta, args.save_freq, args.outfile, args.verbose)


















