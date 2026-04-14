import itertools
from matplotlib import pyplot as plt
import numpy as np

class RealNumber:
    """
    Base class for representing real numbers, dispatching to a specific binary representation subclass.
    The specific representation is determined by the 'binary_representation' argument passed to the constructor.
    The base class itself is not meant to be instantiated directly, but rather serves as a factory for creating
    instances of the appropriate subclass based on the specified binary representation.
    """
    def __new__(cls, n_qubits, binary_representation, a_min=None, a_max=None):
        if cls is RealNumber:
            registry = {
                'real': Real,
                'real_reduced': RealReduced,
                'real_positive': RealPositive,
                'normalized': Normalized,
                'range': Range,
            }
            subcls = registry.get(binary_representation)
            if subcls is None:
                raise Exception('Unknown binary representation', binary_representation)
            obj = super().__new__(subcls)  # create instance of subclass
            subcls.__init__(obj, n_qubits, binary_representation, a_min, a_max)  # initialize subclass
            obj.binary_representation = binary_representation
            return obj
        # If called on a subclass directly, proceed normally
        return super().__new__(cls)

    # Keep the original signature; not used in practice on the base.
    def __init__(self, n_qubits, binary_representation, a_min=None, a_max=None):
        self.n_qubits = n_qubits
        self.binary_representation = binary_representation

    # Keep the original signature for compatibility.
    def evaluate(self, q):
        raise NotImplementedError("Subclasses must implement evaluate().")

    def get_all_possible_values(self, visualize=False, verbose=False):

        binary_combinations = list(itertools.product([0, 1], repeat=self.n_qubits))
        if verbose:
            print('Number of possible binary combinations:', len(binary_combinations))

        values = []
        for q in binary_combinations:
            values.append(self.evaluate(q))
        values = np.sort(values)
        unique_values = np.unique(values)
        distances = np.abs(np.diff(values))
        if verbose:
            print(f'Number of unique values: {len(unique_values)} ({len(unique_values)/len(binary_combinations)*100:.2f}%)')
            print(f'Unique values: min={np.min(unique_values)}, max={np.max(unique_values)}')
            print(f'Distance between values: min={np.min(distances)}, max={np.max(distances)}')
        if visualize:
            plt.figure()
            plt.stem( unique_values, markerfmt='')
            plt.title(f'All unique values for "{self.binary_representation}" representation with {self.n_qubits} qubits')
            plt.show()

        return unique_values

    def find_best_approximation(self, value_to_approximate):
        binary_combinations = list(itertools.product([0, 1], repeat=self.n_qubits))
        diff_opt = np.inf
        q = None
        for q in binary_combinations:
            value = self.evaluate(q)
            diff = np.abs(value-value_to_approximate)
            if diff < diff_opt:
                diff_opt = diff
                q_opt = q

        diff_opt_rel = diff_opt/np.abs(value_to_approximate)

        print('Value:', value_to_approximate,\
             'Approximation:', self.evaluate(q_opt),\
             'Binary:', q_opt,\
             'Difference:', diff_opt, \
             'Relative Difference', diff_opt_rel)

        return q_opt, diff_opt, diff_opt_rel

class Real(RealNumber):
    def __init__(self, n_qubits, binary_representation, a_min=None, a_max=None):
        self.n_qubits = n_qubits
        if not ((self.n_qubits % 2) == 0 and self.n_qubits >= 6):
            raise Exception('Number of qubits has to be even and >=6 for real representation.')

    def evaluate(self, q):
        assert len(q) == self.n_qubits
        m = (self.n_qubits - 2) // 4
        a = -sum(2**(l - m) * q[l] for l in range(2*m + 1)) \
            + sum(2**(l - 3*m - 1) * q[l] for l in range(2*m + 1, 4*m + 2))
        return a

class RealReduced(RealNumber):
    def __init__(self, n_qubits, binary_representation, a_min=None, a_max=None):
        self.n_qubits = n_qubits
        if not ((self.n_qubits % 2) == 0 and self.n_qubits >= 4):
            raise Exception('Number of qubits has to be even and >=4 for real_reduced representation.')

    def evaluate(self, q):
        assert len(q) == self.n_qubits
        m = (self.n_qubits - 2) // 2
        a = -2**m + sum(2**(l - m - 1) * q[l] for l in range(1, 2*m + 2))
        return a

class RealPositive(RealNumber):
    def __init__(self, n_qubits, binary_representation, a_min=None, a_max=None):
        self.n_qubits = n_qubits
        if not ((self.n_qubits % 2) == 1 and self.n_qubits >= 3):
            raise Exception('Number of qubits has to be odd and >=3 for real_positive representation.')

    def evaluate(self, q, a_min=None, a_max=None):
        assert len(q) == self.n_qubits
        m = (self.n_qubits - 1) // 2
        a = sum(2**(l - m) * q[l] for l in range(2*m + 1))
        return a

class Normalized(RealNumber):
    def __init__(self, n_qubits, binary_representation, a_min=None, a_max=None):
        self.n_qubits = n_qubits

    def evaluate(self, q, a_min=None, a_max=None):
        assert len(q) == self.n_qubits
        a = sum(2**l / (2**self.n_qubits - 1) * q[l] for l in range(self.n_qubits))
        return a

class Range(RealNumber):
    def __init__(self, n_qubits, binary_representation, a_min=None, a_max=None):
        self.n_qubits = n_qubits
        self.a_min = a_min
        self.a_max = a_max
        if self.a_min is None or self.a_max is None:
            raise Exception('a_min and a_max must be set for range representation.')
        if self.a_max <= self.a_min:
            raise Exception('a_max must be greater than a_min.')

    def evaluate(self, q):
        assert len(q) == self.n_qubits

        F = 2 ** np.arange(self.n_qubits, dtype=float)
        scale = (self.a_max - self.a_min) / (2**self.n_qubits - 1)
        return self.a_min + scale * sum(F[l] * q[l] for l in range(self.n_qubits))

    def set_range(self, a_min, a_max):
        if a_max <= a_min:
            raise Exception('a_max must be greater than a_min.')
        self.a_min = a_min
        self.a_max = a_max

    def update_range(self, a, a_bit_array, a_prev, a_min, a_max, relaxation_factor):

        actions = []
        c = relaxation_factor

        old_min = a_min
        old_max = a_max

        delta = a_max - a_min
        delta_min = a_prev - a_min
        delta_max = a_max - a_prev

        if a < a_prev:
            a_max = a_max - c*delta_max
            print("\tLowering upper bound.")
            actions.append("max ▼")
        elif a > a_prev:
            a_min = a_min + c*delta_min
            print("\tRaising lower bound.")
            actions.append("min ▲")
        elif a == a_prev:
            a_min = a - delta * c/4
            a_max = a + delta * c/4
            print("\tShrinking range.")
            actions.append("shrink ▶◀")

        # Expand range if all qubits at a node are 1 or 0
        if all(b == 1 for b in a_bit_array):
            print(f"\tAll qubits are 1, expanding range in positive direction.")
            a_max += 0.25 * delta
            actions.append("max ▲")
        elif all(b == 0 for b in a_bit_array):
            print(f"\tAll qubits are 0, expanding range in negative direction.")
            a_min -= 0.25 * delta
            actions.append("min ▼")

        if old_min != a_min or old_max != a_max:
            print(f"\tUpdated range: [{a_min}, {a_max}], ({(a_max - a_min):.6e})")
        else:
            actions.append("none")

        return a_min, a_max, actions

    def re_encode_number(self, a, a_min_new, a_max_new):

        a_bits = self.encode_real_to_bits(a, a_min_new, a_max_new)
        a_new = self.decode_bits_to_real(a_bits, a_min_new, a_max_new)
        rel_err_a_encoded = np.abs(a_new - a) / np.abs(a) if a != 0 else np.nan

        return a_bits, a_new, rel_err_a_encoded

    def encode_real_to_bits(self,value, a_min, a_max):
        if a_max == a_min:
            norm = 0.0
        else:
            norm = (value - a_min) / (a_max - a_min)
        norm = min(max(norm, 0.0), 1.0)
        int_val = int(round(norm * (2**self.n_qubits - 1)))
        bits = [int(c) for c in format(int_val, f'0{self.n_qubits}b')[::-1]]
        return bits

    def decode_bits_to_real(self, bits, a_min, a_max):
        int_val = sum((b & 1) << i for i, b in enumerate(bits))
        denom = (2**self.n_qubits - 1)
        norm = int_val / denom if denom > 0 else 0.0
        value = a_min + norm * (a_max - a_min)
        return value




