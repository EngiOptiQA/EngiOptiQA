import itertools
from matplotlib import pyplot as plt
import numpy as np

class RealNumber:
    def __init__(self, n_qubits, representation):
        self.n_qubits = n_qubits 
        self.representation = representation

    # Binary representation of a real numbers.
    def evaluate(self,q):
        assert(len(q) == self.n_qubits)
        # print('n_qubits = ', len(q))
        # print('logical qubits = ', q)
        if self.representation == 'real':
            if (self.n_qubits % 2) == 0 and self.n_qubits>=6:
                m = int((self.nqubits - 2)/4)
                # print('m = ', m)
                a = -sum([2**(l-m) * q[l] for l in range(2*m+1)]) + sum([2**(l-3*m-1) * q[l] for l in range(2*m+1, 4*m+2)])
                # print('a = ', a)
            else:
                raise Exception('Number of qubits has to be even and >=6 \
                                for real representation.')
        elif self.representation == 'real_reduced':
            if (self.n_qubits % 2) == 0 and self.n_qubits>=4:
                m = int((self.n_qubits - 2)/2)
                # print('m = ', m)
                a = -2**(m) + sum([2**(l-m-1) * q[l] for l in range(1, 2*m+2)])
                # print('a = ', a)
            else:
                raise Exception('Number of qubits has to be even and >=4 \
                                for real_reduced representation.')
        elif self.representation == 'real_positive': 
            if (self.n_qubits % 2) == 1 and self.n_qubits>=3:
                m = int((self.n_qubits - 1)/2)
                # print('m = ', m)
                a = sum([2**(l-m) * q[l] for l in range(2*m+1)])
                # print('a = ', a)
            else:
                raise Exception('Number of qubits has to be odd and >=3 \
                                for real_positive representation.')
        elif self.representation == 'normalized':
            # print('nqubits = ', nqubits)
            # print('q = ', q)
            # a = []
            # for l in range(nqubits):
            #     print(2**l/(2**nqubits-1) * q[l])
            #     a.append(2**l/(2**nqubits-1) * q[l])
            # print(sum([2**l/(2**nqubits-1) * q[l] for l in range(nqubits)]))
            a = sum([2**l/(2**self.n_qubits-1) * q[l] for l in range(self.n_qubits)])
            # print('a = ', a)
        else:
            raise Exception('Unknown representation', self.representation)
        # print('a =', a)
        return a
    
    def plot_all_possible_values(self):
        
        binary_combinations = list(itertools.product([0, 1], repeat=self.n_qubits))
        print('Number of possible values:', len(binary_combinations))
        values = []
        for q in binary_combinations:
            values.append(self.evaluate(q))
        values = np.sort(values)
        distances = np.abs(np.diff(values))
        print(distances)
        #x = np.arange(len(values))
        #print()
        plt.figure()
        plt.scatter(values,values)
        
        plt.show()

        plt.figure()
        plt.plot(distances)
        plt.show()

    def find_best_approximation(self, value_to_approximate):
        binary_combinations = list(itertools.product([0, 1], repeat=self.n_qubits))
        diff_opt = np.Inf
        q = None
        for q in binary_combinations:
            value = self.evaluate(q)
            diff = np.abs(value-value_to_approximate) 
            if diff < diff_opt:
                diff_opt = diff
                q_opt = q

        diff_opt_rel = diff_opt/np.abs(value_to_approximate)
        #print('Value:', value_to_approximate,\
        #      'Approximation:', self.evaluate(q_opt),\
        #      'Difference:', diff_opt, \
        #      'Relative Difference', diff_opt_rel)
        return diff_opt, diff_opt_rel