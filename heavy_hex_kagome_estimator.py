import numpy as np
import random
import math
from qiskit import QuantumCircuit

class HeavyHexKagomeEstimator:
    """
    Custom estimator designed for the XXX Heisenberg hamiltonian on a Kagome lattice
    to be run on a quantum computer based on the Heavy Hex architecture (like the IBMQ
    Guadalupe).
    It uses 2 error mitigation mechnisms:
    - Twirled REadout eXtinction algorithm (T-REx)
    - Circuit error mitigation based on control qubits
    """
    
    def __init__(self, device_edges, kagome_triangles):
        """
        Create an instance of the estimator.
        Caution: this function does NOT check that the kagome lattice, the device and the mapping are correct

        Args:
            device_edges (List[Tuple[int]]): list of connected qubits on the device
            kagome_triangles (List[dict]): list of kagome triangle with a possible control qubit (e.g. {"kagome": (0, 2, 4), "control": 1})
        """

        # Save the arguments
        self.device_edges = device_edges
        self.kagome_triangles = kagome_triangles
        # Compute the number of physical qubits, and 
        self.qubit_count = np.max(self.device_edges) + 1
        # Compute the list of Kagome edges and vertices (used to compute the energy)
        self.kagome_edges = []
        self.kagome_vertices = []
        for triangle in self.kagome_triangles:
            self.kagome_edges.append([triangle["kagome"][0], triangle["kagome"][1]])
            self.kagome_edges.append([triangle["kagome"][1], triangle["kagome"][2]])
            self.kagome_edges.append([triangle["kagome"][2], triangle["kagome"][0]])
            for vertex in triangle["kagome"]:
                if vertex not in self.kagome_vertices:
                    self.kagome_vertices.append(vertex)
    
    def qiskit_number_to_bit_array(self, qiskit_number):
        """
        Convert a number returned by the qiskit sampler primitive to a tuple containing '0' or '1'
        for each qubit. Qiskit uses little-endian convention, hence the use of reversed.
        Example: 3 will be converted to ('1', '1', '0', '0', '0', '0') in a 6-qubit context.

        Args:
            qiskit_number(int): the number returned as a key of a qiskit sampler result

        Returns:
            bitstring (tuple[str]): the tuple that we can use to easily get the value of a qubit
        """
        pattern = "{0:0" + str(self.qubit_count) + "b}" # generates "{0:016b}" for qubit_count=16
        return tuple(reversed(list(pattern.format(int(qiskit_number)))))

    def post_process_qiskit_quasi_dists(self, quasi_dists):
        """
        Convert a number returned by the qiskit sampler primitive to a tuple containing '0' or '1'
        for each qubit. Qiskit uses little-endian convention, hence the use of reversed.
        Example: 3 will be converted to ('1', '1', '0', '0', '0', '0') in a 6-qubit context.

        Args:
            qiskit_number(int): the number returned as a key of a qiskit sampler result

        Returns:
            quasi_dists (List[dict]): the quasi_dists with converted keys
        """
        for i, quasi_dist in enumerate(quasi_dists):
            quasi_dists[i] = {self.qiskit_number_to_bit_array(qiskit_number):quasi_dist[qiskit_number] for qiskit_number in quasi_dist.keys()}
        return quasi_dists


    def TREx_f(self, gatestrings, quasi_dists, s):
        """
        Compute the f function of the T-REx algorithm (see https://arxiv.org/abs/2012.09738)
        Note that the probabilities don't have to sum up to 1.0 for each circuit

        Args:
            gatestrings (List[tuple(str)]): a list of tuples like ('I', 'X', ..., 'I') indicating whether there is an X gate or not in the circuit
            quasi_dists (List[dict]): for each circuit, the result of the sampler
            s (tuple(str)): a tuple like ('I', 'Z', 'Z', ..., 'I', 'I') indicating the targeted pauli hamiltonian

        Returns:
            gatestrings (List[tuple(str)]): the list of circuits as a tuple indicating if inserting an X gate or not
        """

        # Instead of working with D as in the paper, we'll use probabilities
        result = 0.0
        probs_sum = 0.0
        # For each circuit:
        for q_idx in range(len(gatestrings)):
            q = gatestrings[q_idx] # same q as in the paper
            # Compute \gamma_{s,q} as in the paper, i.e. checking if q and s commute:
            gamma_sq_ct = 0
            for i in range(len(s)):
                if s[i] == 'I' or q[i] == 'I' or s[i] == q[i]:
                    gamma_sq_ct += 0
                else:
                    gamma_sq_ct += 1
            # Use the definition of \gamma_{s,q} (1 if q and s commute, -1 otherwise)
            gamma_sq = -1
            if gamma_sq_ct % 2 == 0:
                gamma_sq = 1
            # Loop over x, the measurements of the qubits:
            for x in quasi_dists[q_idx].keys():
                # compute <s,x>
                sx = 0
                for i in range(self.qubit_count):
                    if s[i] == 'Z' and x[i] == '1':
                        sx += 1
                # add probability * \gamma_{s,q} * (-1)^{<s,x>}
                prob = quasi_dists[q_idx][x]
                result += prob * gamma_sq * (-1)**sx
                probs_sum += prob
        if probs_sum == 0.0:
            return 0.0 # In order to avoid an exception in special cases such as noiseless environments
        return result / probs_sum
    

    def generate_balanced_random_X_gates(self, num_twirled_circuits, where_to_apply):
        """
        Generates a list of twirled circuits X instructions.
        In order to ensure both 0s and 1s well distributed, for each qubit, we take randomly
        half of circuits and set a X gate, keeping the other half unchanged.
        Example:
        For 3 qubits, num_twirled_circuits=2 and where_to_apply=[0], it will return:
        [('X', 'I', 'I'), ('I', 'I', 'I')]
        or 
        [('I', 'I', 'I'), ('X', 'I', 'I')]
        with half of the circuits having an X gate for the qubit #0

        Args:
            num_twirled_circuits (int): number of twirled circuits to use
            where_to_apply (List[int]): the list of qubits where we want to apply randomly an X gate

        Returns:
            gatestrings (List[tuple(str)]): the list of circuits as a tuple indicating if inserting an X gate or not
        """
        gatestrings = []
        for i in range(num_twirled_circuits):
            gatestrings.append(['I' for i in range(self.qubit_count)])
        gatestring_indices = [i for i in range(num_twirled_circuits)]
        for i in where_to_apply:
            random.shuffle(gatestring_indices)
            for k in range(num_twirled_circuits//2):
                gatestrings[gatestring_indices[k]][i] = 'X'
        return gatestrings


    def calibrate(self, sampler, num_twirled_circuits, shots):
        """
        Calibrate the estimator

        Args:
            sampler: the sampler to use
            num_twirled_circuits (int): number of twirled circuits to use
            shots (int): number of shots per circuit to use
        """

        # 1. Prepare calibration circuits with, for each qubit, either an X gate or no gate
        # Let's prepare when apply X gates for all qubits
        gatestrings = self.generate_balanced_random_X_gates(num_twirled_circuits, [i for i in range(self.qubit_count)])
        # Use the generated gate strings to generate actual circuits
        calib_circs = []
        for i in range(num_twirled_circuits):
            # Create a circuit
            circ = QuantumCircuit(self.qubit_count, self.qubit_count)
            # Choose randomly for each qubit if applying an X gate or not
            gatestring = gatestrings[i]
            # Add a X gate when necessary
            for qubit in range(self.qubit_count):
                if gatestring[qubit] == 'X':
                    circ.x(qubit)
            # Measure all qubits
            for qubit in range(self.qubit_count):
                circ.measure(qubit, qubit)
            # Add the circuit
            calib_circs.append({
                "circuit": circ,
                "gatestring": gatestring
            })
        # 2. Execute the circuits with the sampler primitive
        job = sampler.run([el["circuit"] for el in calib_circs], [[] for el in calib_circs], shots=shots)
        quasi_dists = self.post_process_qiskit_quasi_dists(job.result().quasi_dists)
        # 3. For all the kagome edges, compute the T-REx calibration value that will make it possible to 
        # later estimate the edge hamiltonian energies
        self.trex_edge_calibrations = {} # key: the edge, value: the result of the f function for the T-REx calibration
        for edge in self.kagome_edges:
            # Prepare s (as in the T-REx paper) for this edge, i.e. the Z-pauli measurement hamiltonian
            s = ['I' for i in range(self.qubit_count)]
            s[edge[0]] = 'Z'
            s[edge[1]] = 'Z'
            fvalue = self.TREx_f([el["gatestring"] for el in calib_circs], quasi_dists, s)
            self.trex_edge_calibrations[(edge[0], edge[1])] = fvalue
            self.trex_edge_calibrations[(edge[1], edge[0])] = fvalue
        # 4. For all the control qubits, estimate the probability of the qubit being 0/1 given a measurement of 0/1
        # under uniform apriori distribution
        self.control_qubit_measuring_probs = {} # key: qubit number, value: dictionary of probability (see below)
        for triangle in self.kagome_triangles:
            # If there is no control qubit in the triangle, just ignore it
            if triangle["control"] == None:
                continue
            # Compute the likelihood probabilities (probability of measuring 0/1 given a ground-truth 0/1)
            # ("p" as Probability, "t" as True value, "m" as Measured value)
            pm0_t0 = 0.0 # 0 given a ground-truth of 0
            pm1_t0 = 0.0
            pm0_t1 = 0.0
            pm1_t1 = 0.0
            for circ_idx in range(len(calib_circs)):
                for measure in quasi_dists[circ_idx].keys():
                    if measure[triangle["control"]] == '0': # measuring 0
                        if calib_circs[circ_idx]["gatestring"][triangle["control"]] == 'I': # ground-truth is 0
                            pm0_t0 += quasi_dists[circ_idx][measure]
                        else: # ground-truth is 1
                            pm0_t1 += quasi_dists[circ_idx][measure]
                    else: # measuring 1
                        if calib_circs[circ_idx]["gatestring"][triangle["control"]] == 'I': # ground-truth is 0
                            pm1_t0 += quasi_dists[circ_idx][measure]
                        else: # ground-truth is 1
                            pm1_t1 += quasi_dists[circ_idx][measure]
            # Normalize in order to have pm0_t0+pm1_t0=1.0 and pm0_t1+pm1_t1=1.0
            p_t0 = pm0_t0+pm1_t0
            pm0_t0 /= p_t0
            pm1_t0 /= p_t0
            p_t1 = pm0_t1+pm1_t1
            pm0_t1 /= p_t1
            pm1_t1 /= p_t1
            # Deriving the probabilities under uniform apriori distribution
            self.control_qubit_measuring_probs[triangle["control"]] = {
                "pt0_m1": pm1_t0/(pm1_t0+pm0_t0), # probability that the qubit has a true value 0 given a measurement of 1
                "pt1_m0": pm0_t1/(pm0_t1+pm1_t1) # probability that the qubit has a true value 1 given a measurement of 0
            }
    

    def estimate_energy_for_circuit(self, base_circuit, sampler, num_twirled_circuits, shots):
        """
        Estimates the energy (expectaction value of the hamiltonian) for a circuit.

        Args:
            base_circuit: a quantum circuit
            sampler: the sampler to use
            num_twirled_circuits (int): number of twirled circuits to use
            shots (int): number of shots per circuit to use

        Returns:
            result (dict): a dictionary containing the value of the energy ("energy" key) and other information
        """

        # 1. Let's build a list of circuits to use with the Sampler primitive
        # We want circuits for the X, Y, and the Z part of the hamiltonian, 
        # and randomly apply X gate before measurement (for the T-REx readout error mitigation algorithm)
        circuits = []
        # For X, Y or Z:
        for xyz in ["X", "Y", "Z"]:
            circ_xyz = base_circuit.copy()
            # Let's generate random X gates (for kagome vertices only):
            gatestrings = self.generate_balanced_random_X_gates(num_twirled_circuits, self.kagome_vertices)
            # Add the basis change to all the kagome vertices for X, Y or Z measurement
            if xyz == "X":
                circ_xyz.rz(math.pi/2.0, self.kagome_vertices)
                circ_xyz.sx(self.kagome_vertices)
                circ_xyz.rz(math.pi/2.0, self.kagome_vertices)
            elif xyz == "Y":
                circ_xyz.sx(self.kagome_vertices)
                circ_xyz.rz(math.pi/2.0, self.kagome_vertices)
            elif xyz == "Z":
                pass # nothing to do for Z measurement
            # And then we apply the X gates (for T-REx) and measuring
            for gatestring in gatestrings:
                circ = circ_xyz.copy()
                # Add X gates:
                for qubit in range(self.qubit_count):
                    if gatestring[qubit] == 'X':
                        circ.x(qubit)
                # Add measurements:
                circ.measure([i for i in range(self.qubit_count)], [i for i in range(self.qubit_count)])
                # Add the circuit
                circuits.append({
                    "circuit": circ,
                    "xyz": xyz, 
                    "gatestring": gatestring
                })
                
        # 2. Execute the circuits with the sampler primitive
        job = sampler.run([el["circuit"] for el in circuits], [[] for el in circuits], shots=shots)
        quasi_dists = self.post_process_qiskit_quasi_dists(job.result().quasi_dists)
        for i, quasi_dist in enumerate(quasi_dists):
            circuits[i]["quasi_dist"] = quasi_dist # add the results to the circuits variable
        
        # 3. For each triangle, and for each X,Y,Z, let's estimate the energy
        triangle_results = []
        for xyz in ["X", "Y", "Z"]:
            xyz_circuits = [el for el in circuits if el["xyz"] == xyz] # keep the results corresponding to xyz
            # For each triangle:
            for triangle_idx, triangle in enumerate(self.kagome_triangles):
                # List the kagome edges belonging to the triangle:
                triangle_edges = [
                    (triangle["kagome"][0], triangle["kagome"][1]),
                    (triangle["kagome"][1], triangle["kagome"][2]),
                    (triangle["kagome"][2], triangle["kagome"][0])
                ]
                # First, compute the triangle energy with just the T-REx algorithm 
                # without the control-qubit error mitigation (as a comparison)
                triangle_energy_without_control = 0.0
                for edge in triangle_edges:
                    # Prepare s (as in the T-REx paper) for this edge, i.e. the Z-pauli measurement hamiltonian
                    s = ['I' for i in range(self.qubit_count)]
                    s[edge[0]] = 'Z'
                    s[edge[1]] = 'Z'
                    # Estimate the edge energy, as in the T-REx paper
                    edge_energy = self.TREx_f([el["gatestring"] for el in xyz_circuits], [el["quasi_dist"] for el in xyz_circuits], s) / self.trex_edge_calibrations[edge]
                    # add the edge energy to the triangle energy
                    triangle_energy_without_control += edge_energy
                
                # If the triangle has no control qubit, we just use the T-Rex algorithm
                if triangle["control"] == None:
                    triangle_energy = triangle_energy_without_control
                # If the triangle has a control qubit:
                else:
                    # Compute p0: frequency of measuring 0 for the control qubit
                    p0 = 0.0
                    for circ in xyz_circuits:
                        for measure in circ["quasi_dist"].keys():
                            if measure[triangle["control"]] == '0': # if 0 is measured
                                p0 += circ["quasi_dist"][measure] # then we add the associated probability
                    p0 /= len(xyz_circuits) # there are several circuits, we need to normalize so that p0+p1=1
                    # Compute p1: frequency of measuring 1 for the control qubit
                    p1 = 1.0 - p0
                    # Separate the results of quasi_distributions depending on the measure on the control qubit:
                    quasi_dists0 = [{k:circ["quasi_dist"][k] for k in circ["quasi_dist"] if k[triangle["control"]] == '0'} for circ in xyz_circuits]
                    quasi_dists1 = [{k:circ["quasi_dist"][k] for k in circ["quasi_dist"] if k[triangle["control"]] == '1'} for circ in xyz_circuits]
                    # Compute E0: the energy associated with measuring 0 for the control qubit
                    E0 = 0.0
                    for edge in triangle_edges:
                        # Prepare s (as in the T-Rex paper) for this edge, i.e. the Z-pauli measurement hamiltonian
                        s = ['I' for i in range(self.qubit_count)]
                        s[edge[0]] = 'Z'
                        s[edge[1]] = 'Z'
                        # Estimate the edge energy, as in the T-Rex paper
                        edge_energy = self.TREx_f([el["gatestring"] for el in xyz_circuits], quasi_dists0, s) / self.trex_edge_calibrations[edge]
                        # add the edge energy to the triangle energy
                        E0 += edge_energy
                    # Compute E1: the energy associated with measuring 1 for the control qubit
                    E1 = 0.0
                    for edge in triangle_edges:
                        # Prepare s (as in the T-Rex paper) for this edge, i.e. the Z-pauli measurement hamiltonian
                        s = ['I' for i in range(self.qubit_count)]
                        s[edge[0]] = 'Z'
                        s[edge[1]] = 'Z'
                        # Estimate the edge energy, as in the T-Rex paper
                        edge_energy = self.TREx_f([el["gatestring"] for el in xyz_circuits], quasi_dists1, s) / self.trex_edge_calibrations[edge]
                        # add the edge energy to the triangle energy
                        E1 += edge_energy
                
                    # Apply a correction by taking into account readout errors on the control qubits
                    # and find corrected values p0c, p1c, E0c, E1c ("c" as corrected)
                    # Retrieve the probability of the control qubit
                    pt0_m1 = self.control_qubit_measuring_probs[triangle["control"]]["pt0_m1"] # probability of being 0 given that 1 is measured
                    pt1_m0 = self.control_qubit_measuring_probs[triangle["control"]]["pt1_m0"] # probability of being 1 given that 0 is measured
                    # The equations for correction are:
                    # E0 = (1-pt1_m0) * E0c + pt1_m0 * E1c
                    # E1 = (1-pt0_m1) * E1c + pt0_m1 * E0c
                    # p0 = (1-pt1_m0) * p0c + pt1_m0 * p1c
                    # p1 = (1-pt0_m1) * p1c + pt0_m1 * p0c
                    # Let's derive the corrected values (equivalent to inverting a 2x2 matrix)
                    E1c = ((1-pt1_m0)*E1 - pt0_m1*E0)/(1-pt0_m1-pt1_m0)
                    p1c = ((1-pt1_m0)*p1 - pt0_m1*p0)/(1-pt0_m1-pt1_m0)
                    E0c = ((1-pt0_m1)*E0 - pt1_m0*E1)/(1-pt0_m1-pt1_m0)
                    p0c = ((1-pt0_m1)*p0 - pt1_m0*p1)/(1-pt0_m1-pt1_m0)
                
                    # Find the estimation of the targeted energy
                    triangle_energy = (p0c * E0c - p1c * E1c) / (p0c-p1c)
                
                # Save results:
                triangle_results.append({
                    "xyz": xyz, 
                    "triangle_idx": triangle_idx,
                    "triangle_energy": triangle_energy,
                    "triangle_energy_without_control": triangle_energy_without_control
                })
                
        # 4. Return the results:
        return {
            "energy": np.sum([triangle_result["triangle_energy"] for triangle_result in triangle_results]),
            "energy_without_control": np.sum([triangle_result["triangle_energy_without_control"] for triangle_result in triangle_results]),
            "triangle_results": triangle_results
        }