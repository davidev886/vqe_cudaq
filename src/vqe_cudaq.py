import cudaq


class VqeHardwareEfficient(object):
    def __init__(self, n_qubits, n_layers, n_electrons=None):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.num_params = n_qubits * (n_layers + 1)
        self.final_state_vector_best = None
        self.best_vqe_params = None
        self.best_vqe_energy = None
        self.n_electrons = n_electrons

    def layers(self):
        n_qubits = self.n_qubits
        n_layers = self.n_layers

        kernel, thetas = cudaq.make_kernel(list)
        # Allocate n qubits.
        qubits = kernel.qalloc(n_qubits)
        # kernel.x(qubits[0])
        # Apply an `ry` gate that's parameterized by the first
        # `QuakeValue` entry of our list, `thetas`.
        # kernel.ry(thetas[0], qubits[1])
        # kernel.cx(qubits[1], qubits[0])
        # Note: the kernel must not contain measurement instructions.
        # layer 0
        i = 0
        for q in range(n_qubits):
            kernel.ry(thetas[i * n_qubits + q], qubits[q])
            kernel.rz(thetas[i * n_qubits + q], qubits[q])

        # layers 1, ..., p
        for i in range(n_layers):
            for q in range(n_qubits-1):
                kernel.cx(qubits[q], qubits[q + 1])

            for q in range(n_qubits):
                kernel.ry(thetas[i * n_qubits + q], qubits[q])
                kernel.rz(thetas[i * n_qubits + q], qubits[q])

        return kernel, thetas

    def run_vqe_cudaq(self, hamiltonian, options={}):
        # cudaq.set_qpu('qpp')
        optimizer = cudaq.optimizers.COBYLA()
        kernel, thetas = self.layers()
        optimizer.max_iterations = options.get('maxiter', 10)

        # optimizer...

        # Finally, we can pass all of that into `cudaq.vqe` and it will automatically run our
        # optimization loop and return a tuple of the minimized eigenvalue of our `spin_operator`
        # and the list of optimal variational parameters.
        energy, parameter = cudaq.vqe(
            kernel=kernel,
            spin_operator=hamiltonian,
            optimizer=optimizer,
            parameter_count=self.num_params)

        print(energy, parameter)
