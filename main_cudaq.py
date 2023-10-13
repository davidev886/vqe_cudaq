import numpy as np

#from pyscf.scf.chkfile import dump_scf

from openfermion.transforms import get_fermion_operator, jordan_wigner
from openfermion.chem import MolecularData
#from openfermionpyscf import run_pyscf

from src.run_pyscf import run_pyscf

from src.utils_vqe import write_trial_ipie, write_hamiltonian_ipie
from src.vqe_cudaq_qnp import VqeQnp
from src.utils_cudaq import get_cudaq_hamiltonian

if __name__ == "__main__":
    np.set_printoptions(precision=6, suppress=True, linewidth=10000)
    writing_files = False
    np.random.seed(12)
    basis = "sto-3g"
    multiplicity = 1
    charge = 0
    distance = 1.23
    geometry = [('H', (0, 0, 0)),
                ('H', (0, 0, distance)),
                ('H', (distance, 0, 0)),
                ('H', (distance, 0, distance))]

    basis = 'sto-3g'
    spin = 0

    molecule = MolecularData(geometry, basis, multiplicity, charge)

    molecule = run_pyscf(molecule, verbose=True)

    mf = molecule._pyscf_data['scf']
    mol = molecule._pyscf_data['mol']
    noccas, noccbs = mol.nelec

    init_mo_occ = mf.mo_occ
    print(init_mo_occ)

    print(f"SCF energy: {molecule.hf_energy}")

    n_qubits = molecule.n_qubits
    n_electron = molecule.n_electrons
    fermionic_hamiltonian = get_fermion_operator(molecule.get_molecular_hamiltonian())
    jw_hamiltonian = jordan_wigner(fermionic_hamiltonian)

    hamiltonian_cudaq = get_cudaq_hamiltonian(jw_hamiltonian)
    vqe = VqeQnp(n_qubits=n_qubits,
                 n_layers=10,
                 n_electrons=mol.nelec,
                 init_mo_occ=init_mo_occ)
    energy, params = vqe.run_vqe_cudaq(hamiltonian_cudaq, options={'maxiter': 10000, 'callback': True})

    print(energy, params)
    if writing_files:
        write_hamiltonian_ipie(mf, file_name="hamiltonian.h5")
        write_trial_ipie(vqe.final_state_vector_best, mol.nelec, file_name="wavefunction.h5")
        write_json_input_file(input_filename="ipie_input.json",
                              hamil_filename="hamiltonian.h5",
                              wfn_filename="wavefunction.h5",
                              nelec=mol.nelec)

        # run ipie with
        #       ipie ipie_input.json
