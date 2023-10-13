import numpy as np
from openfermion.transforms import jordan_wigner
from openfermion import generate_hamiltonian
from pyscf import gto, scf, ao2mo, mcscf
from pyscf.lib import chkfile
# from pyscf.scf.chkfile import dump_scf
from src.vqe_cudaq_qnp import VqeQnp
from src.utils_cudaq import get_cudaq_hamiltonian


def molecule_data(atom_name):
    # table B1 angstrom
    molecules = {'ozone': [('O', (0.0000000, 0.0000000, 0.0000000)),
                            ('O', (0.0000000, 0.0000000, 1.2717000)),
                            ('O', (1.1383850, 0.0000000, 1.8385340))]}

    return molecules[atom_name]


if __name__ == "__main__":
    np.set_printoptions(precision=6, suppress=True, linewidth=10000)
    n_vqe_layers = 4
    writing_files = False
    np.random.seed(12)

    geometry = molecule_data('ozone')
    basis = 'cc-pVdZ'
    spin = 0
    multiplicity = spin + 1
    mol = gto.M(
        atom=geometry,
        basis=basis,
        spin=spin,
        verbose=8,
    )
    hf = scf.ROHF(mol)
    hf.kernel()

    print("HF Energy: ", hf.e_tot)

    num_active_orbitals = 6
    num_active_electrons = 4

    my_casci = mcscf.CASCI(hf, num_active_orbitals, num_active_electrons)
    my_casci.kernel()

    ecas = my_casci.kernel()

    print('FCI Energy in CAS:', ecas[0])

    h1, energy_core = my_casci.get_h1eff()
    h2 = my_casci.get_h2eff()
    h2_no_symmetry = ao2mo.restore('1', h2, num_active_orbitals)
    tbi = np.asarray(h2_no_symmetry.transpose(0, 2, 3, 1), order='C')

    mol_ham = generate_hamiltonian(h1, tbi, energy_core)
    jw_hamiltonian = jordan_wigner(mol_ham)
    hamiltonian_cudaq = get_cudaq_hamiltonian(jw_hamiltonian)

    n_qubits = 2 * h1.shape[0]

    mc = my_casci
    casdm1, casdm2 = mc.fcisolver.make_rdm12(mc.ci, mc.ncas, mc.nelecas)
    init_mo_occ = np.round(casdm1.diagonal())

    vqe = VqeQnp(n_qubits=n_qubits,
                 n_layers=n_vqe_layers,
                 n_electrons=mol.nelec,
                 init_mo_occ=init_mo_occ)
    energy, params = vqe.run_vqe_cudaq(hamiltonian_cudaq, options={'maxiter': 10000, 'callback': True})

    print(energy, params)
    if writing_files:
        from src.utils_vqe import write_trial_ipie, write_hamiltonian_ipie
        write_hamiltonian_ipie(hf, file_name="hamiltonian.h5")
        write_trial_ipie(vqe.final_state_vector_best, mol.nelec, file_name="wavefunction.h5")
        write_json_input_file(input_filename="ipie_input.json",
                              hamil_filename="hamiltonian.h5",
                              wfn_filename="wavefunction.h5",
                              nelec=mol.nelec)

        # run ipie with
        #       ipie ipie_input.json
