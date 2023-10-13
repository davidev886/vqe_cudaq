import numpy as np
import h5py
import os
from pyscf import ao2mo
# from ipie.utils.from_pyscf import generate_integrals
# from ipie.utils.io import write_hamiltonian, write_wavefunction
# from ipie.utils.io import write_json_input_file
from functools import reduce


def write_hamiltonian_ipie(mf, file_name="hamiltonian.h5", file_path="./", thres=1e-6, verbose=False):
    """
    :param mf: pyscf MF object
    :param file_name: name of the h5 file to create
    :param file_path: path of the h5 file to create
    :param thres: Threshold for Cholesky decomposition
    :returns: Nothing, but stores the hamiltonian as input for ipie
    """
    h1e, chol, enuc = generate_integrals(
        mf.mol, mf.get_hcore(), mf.mo_coeff, chol_cut=thres, verbose=verbose
    )

    write_hamiltonian(
        h1e, chol.copy(), e0=enuc,
        filename=os.path.join(file_path, file_name)
    )


def write_trial_ipie(state_vector, nelec, file_name="wavefunction.h5", file_path="./", ncore_electrons=0, thres=1e-6):
    """
    This function writes an h5 file with the amplitude and the kets from the
    state vector. The file can be used as input for ipie
    :param state_vector: state vector generated from cirq
    :param nelec: tuple (n_electron_alpha, n_electron_beta)
    :param file_name: name of the h5 file to create
    :param file_path: path of the h5 file to create
    :param thres: Threshold for coefficients to keep from VQE wavefunction
    """
    noccas, noccbs = nelec
    coeff, occas, occbs = prep_input_ipie(state_vector, noccas, noccbs, ncore_electrons, thres)

    write_wavefunction((coeff, occas, occbs),
                       os.path.join(file_path, file_name), nelec)


def normal_ordering_swap(orbitals):
    """
    This function normal calculates the phase coefficient (-1)^swaps, where swaps is the number of
    swaps required to normal order a given list of numbers.
    :param orbitals: list of numbers, e.g. orbitals
    :returns: number of required swaps
    """
    count_swaps = 0
    for i in range(len(orbitals)):
        for j in range(i + 1, len(orbitals)):
            if orbitals[i] > orbitals[j]:
                count_swaps += 1

    return count_swaps


def fix_nelec_in_state_vector(final_state_vector, nelec):
    """
    Projects the wave function final_state_vector in the subspace with the fix number of electrons given by nelec
    :param final_state_vector Cirq object representing the state vector from a VQE simulation
    :param nelec (tuple) with n_alpha, n_beta number of electrons
    return: state vector (correctly normalized) with fixed number of electrons
    """
    n_alpha, n_beta = nelec
    n_qubits = int(np.log2(len(final_state_vector)))
    projected_vector = np.array(final_state_vector)
    for decimal_ket, coeff in enumerate(final_state_vector):
        string_ket = bin(decimal_ket)[2:].zfill(n_qubits)
        string_alpha = string_ket[::2]  # alpha orbitals occupy the even positions
        string_beta = string_ket[1::2]  # beta orbitals occupy the odd positions
        alpha_occ = [pos for pos, char in enumerate(string_alpha) if char == '1']
        beta_occ = [pos for pos, char in enumerate(string_beta) if char == '1']
        if (len(alpha_occ) != n_alpha) or (len(beta_occ) != n_beta):
            projected_vector[decimal_ket] = 0.0

    normalization = np.sqrt(np.dot(projected_vector.conj(), projected_vector))
    return projected_vector / normalization


def prep_input_ipie(final_state_vector, noccas, noccbs, ncore_electrons=0, thres=1e-6):
    """
    :param final_state_vector: Cirq object representing the state vector from a VQE simulation
    :param thres: Threshold for coefficients to keep from VQE wavefunction
    :returns: Input for ipie trial: coefficients, list of strings with the corresponding kets
    """
    bin_ind = [np.binary_repr(i, width=int(np.log2(len(final_state_vector)))) for i in
               range(len(final_state_vector))]
    coeffs = []
    occas = []
    occbs = []

    for k, i in enumerate(bin_ind):
        alpha_aux = []
        beta_aux = []
        for j in range(len(i) // 2):
            alpha_aux.append(i[2 * j])
            beta_aux.append(i[2 * j + 1])
        alpha_occ = [i for i, x in enumerate(alpha_aux) if x == '1']
        beta_occ = [i for i, x in enumerate(beta_aux) if x == '1']
        if np.abs(final_state_vector[k]) >= thres:
            coeffs.append(final_state_vector[k])
            occas.append(alpha_occ)
            occbs.append(beta_occ)
    # We need it non_normal ordered
    for i in range(len(coeffs)):
        exponent = normal_ordering_swap([2 * j for j in occas[i]] + [2 * j + 1 for j in occbs[i]])
        coeffs[i] = coeffs[i] * (-1) ** exponent

    ncore = ncore_electrons // 2
    core = [i for i in range(ncore)]
    occas = [np.array(core + [o + ncore for o in oa]) for oa in occas]
    occbs = [np.array(core + [o + ncore for o in ob]) for ob in occbs]

    coeffs = np.array(coeffs, dtype=complex)
    normalization = np.dot(coeffs.conj(), coeffs) ** 0.5
    ixs = np.argsort(np.abs(coeffs))[::-1]
    coeffs = coeffs[ixs] / normalization.real
    occas = np.array(occas, dtype=int)[ixs]
    occbs = np.array(occbs, dtype=int)[ixs]

    return coeffs, occas, occbs


def prep_input_ipie_old(final_state_vector, thres=1e-6):
    """
    :param final_state_vector: Cirq object representing the state vector from a VQE simulation
    :param thres: Threshold for coefficients to keep from VQE wavefunction
    :returns: Input for ipie trial: coefficients, list of strings with the corresponding kets
    """
    n_qubits = int(np.log2(len(final_state_vector)))

    coeffs = []
    kets = []
    for j, coeff in enumerate(final_state_vector):
        if np.abs(coeff) >= thres:
            coeffs.append(coeff)
            kets.append([int(_) for _ in bin(j)[2:].zfill(n_qubits)])

    coeffs = np.array(coeffs, dtype=complex)

    ixs = np.argsort(np.abs(coeffs))[::-1]
    coeffs = coeffs[ixs]
    kets = np.array(kets)[ixs]

    return coeffs, kets


def compute_integrals(pyscf_molecule, pyscf_scf):
    """
    Compute the 1-electron and 2-electron integrals.

    Args:
        pyscf_molecule: A pyscf molecule instance.
        pyscf_scf: A PySCF "SCF" calculation object.

    Returns:
        one_electron_integrals: An N by N array storing h_{pq}
        two_electron_integrals: An N by N by N by N array storing h_{pqrs}.
    """
    # Get one electrons integrals.
    n_orbitals = pyscf_scf.mo_coeff.shape[1]
    one_electron_compressed = reduce(np.dot, (pyscf_scf.mo_coeff.T,
                                              pyscf_scf.get_hcore(),
                                              pyscf_scf.mo_coeff))

    one_electron_integrals = one_electron_compressed.reshape(
        n_orbitals, n_orbitals).astype(float)

    # Get two electron integrals in compressed format.
    two_electron_compressed = ao2mo.kernel(pyscf_molecule,
                                           pyscf_scf.mo_coeff)

    two_electron_integrals = ao2mo.restore(
        1,  # no permutation symmetry
        two_electron_compressed, n_orbitals)
    # See PQRS convention in OpenFermion.hamiltonians._molecular_data
    # h[p,q,r,s] = (ps|qr)
    two_electron_integrals = np.asarray(
        two_electron_integrals.transpose(0, 2, 3, 1), order='C')

    # Return.
    return one_electron_integrals, two_electron_integrals
