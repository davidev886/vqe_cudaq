import numpy as np

#from pyscf.scf.chkfile import dump_scf

from openfermion.transforms import get_fermion_operator, jordan_wigner
from openfermion.chem import MolecularData
#from openfermionpyscf import run_pyscf
from openfermion import generate_hamiltonian

from src.run_pyscf import run_pyscf

#from src.utils_vqe import write_trial_ipie, write_hamiltonian_ipie

from pyscf import gto, scf, ao2mo, mcscf
from pyscf.lib import chkfile

from src.vqe_cudaq_qnp import VqeQnp
from src.utils_cudaq import get_cudaq_hamiltonian

if __name__ == "__main__":
    np.set_printoptions(precision=6, suppress=True, linewidth=10000)
    writing_files = False
    np.random.seed(12)
    chkptfile = "mcscf.chk"
    basis='cc-pVTZ'
    spin = 1
    multiplicity = spin + 1
    charge=0
    atom='geo.xyz'
    mol = gto.M(
        atom=atom,
        spin=spin,
        charge=charge,
        basis=basis,
        verbose=8
        )

    mf = scf.ROHF(mol)
    mf.kernel()

    num_active_orbitals = 5
    num_active_electrons = 5

    my_casci = mcscf.CASCI(mf, num_active_orbitals, num_active_electrons)
    x = (mol.spin/2 * (mol.spin/2 + 1))
    print(f"x={x}")
    my_casci.fix_spin_(ss=x)
    if chkptfile and os.path.exists(chkptfile):
        mo = chkfile.load(chkptfile, 'mcscf/mo_coeff')
        ecas, *_ = my_casci.kernel(mo)
    else:
        ecas, *_ = my_casci.kernel()

    print('FCI Energy in CAS:', ecas)

    h1, energy_core = my_casci.get_h1eff()
    h2 = my_casci.get_h2eff()
    h2_no_symmetry = ao2mo.restore('1', h2, num_active_orbitals)
    tbi = np.asarray(h2_no_symmetry.transpose(0, 2, 3, 1), order='C')

    mol_ham = generate_hamiltonian(h1, tbi, energy_core)
    jw_hamiltonian = jordan_wigner(mol_ham)
    hamiltonian_cudaq = get_cudaq_hamiltonian(jw_hamiltonian)

    n_qubits = 2 * h1.shape[0]

    casdm1, casdm2 = my_casci.fcisolver.make_rdm12(my_casci.ci, my_casci.ncas, my_casci.nelecas)
    init_mo_occ = np.round(casdm1.diagonal())

    noccas, noccbs = mol.nelec
    n_electron = noccas + noccbs
    print("Starting VQE")
    vqe = VqeQnp(n_qubits=n_qubits,
                 n_layers=10,
                 n_electrons=n_electron,
                 init_mo_occ=init_mo_occ)

    energy, params = vqe.run_vqe_cudaq(hamiltonian_cudaq, options={'maxiter': 10000, 'callback': True})

    print("Best energy, Best params")
    print(energy, params)

    if writing_files:
        from src.utils_vqe import write_trial_ipie, write_hamiltonian_ipie
        write_hamiltonian_ipie(mf, file_name="hamiltonian.h5")
        write_trial_ipie(vqe.final_state_vector_best, mol.nelec, file_name="wavefunction.h5")
        write_json_input_file(input_filename="ipie_input.json",
                              hamil_filename="hamiltonian.h5",
                              wfn_filename="wavefunction.h5",
                              nelec=mol.nelec)

        # run ipie with
        #       ipie ipie_input.json
