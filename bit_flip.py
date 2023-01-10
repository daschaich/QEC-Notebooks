import matplotlib.pyplot as plt
from qiskit_aer.noise import pauli_error
from qiskit import *
error_probabilities = []
success_rates = []
for error_probability in range(0,101,1):    
    error_probability = error_probability / 1000.0
    error_probabilities.append(error_probability)
    # Create a quantum register with 5 qubits
    q = QuantumRegister(5)
    
    # Create a classical register with 5 bits
    c = ClassicalRegister(2)
    
    # Create a quantum circuit
    bit_flip = QuantumCircuit(q, c)
    
    # Prepares qubit in the desired initial state 
    
    # Encodes the qubit in a three-qubit entangled state  
    bit_flip.cx(q[0], q[1])
    bit_flip.cx(q[0], q[2])
    
    # Construct a 1-qubit bit-flip error
    p_error = error_probability
    error = pauli_error([('X', p_error/2),('Z', p_error/2), ('I', 1 - p_error)])
    
    
    # Combine the circuit and the error channel
    for i in range(0,3):
        bit_flip.append(error, [i])
    
    # Adds additional two qubits for error-correction
    bit_flip.cx(q[0], q[3])
    bit_flip.cx(q[1], q[3])
    bit_flip.cx(q[1], q[4])
    bit_flip.cx(q[2], q[4])
    
    # Measure the two additional qubits
    bit_flip.measure(q[3], c[0])
    bit_flip.measure(q[4], c[1])
    
    # Do error correction
    bit_flip.x(q[0]).c_if(c,1)
    bit_flip.x(q[1]).c_if(c,3)
    bit_flip.x(q[2]).c_if(c,2)
    
    # Decodes the qubit from the three-qubit entangled state  
    # bit_flip.cx(q[0], q[1])
    # bit_flip.cx(q[0], q[2])
    
    # Create a classical register with 3 bits
    cr = ClassicalRegister(3)
                
    # Add the classical register to the quantum circuit
    bit_flip.add_register(cr)
    # Check the state of the initial qubit
    for i in range(0,3):
        bit_flip.measure(q[i], cr[i])
    # Choose a simulator
    simulator = Aer.get_backend('qasm_simulator')
    
    # Execute the circuit on the simulator
    result = execute(bit_flip, simulator, shots = 1000).result()
    
    # Get the counts from the result
    counts = result.get_counts()
    
    num_successes = 0
    counts_dict = dict(counts)
    for key, value in counts_dict.items():
        if key[:3] == '000':
            num_successes += value
    #print(error_probability,num_successes)
    success_rate = num_successes / 1000.0
    success_rates.append(success_rate)
    
#Plot the success rate as a function of the error probability
plt.plot(error_probabilities, success_rates)
plt.title('Logical0 performance for 3 qubit bit flip code')
plt.xlabel('Error probability')
plt.ylabel('Success rate')
plt.show()
