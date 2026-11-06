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
    _registry = {}

    @classmethod
    def create(cls, n_qubits, binary_representation,
               *, a_min=None, a_max=None,
               a_min_lim=None, a_max_lim=None, mu=None):

        try:
            subcls = cls._registry[binary_representation]
        except KeyError:
            raise ValueError(
                f"Unknown binary representation: {binary_representation!r}"
            ) from None

        if binary_representation in {
            "real", "real_reduced", "real_positive", "normalized"
        }:
            obj = subcls(n_qubits)

        elif binary_representation in {"range_binary", "range_unary"}:
            obj = subcls(n_qubits, a_min=a_min, a_max=a_max)

        elif binary_representation == "range_bce":
            obj = subcls(n_qubits, a_min=a_min, a_max=a_max, mu=mu)

        elif binary_representation in {"adaptive_range_binary", "adaptive_range_unary"}:
            obj = subcls(
                n_qubits,
                a_min=a_min,
                a_max=a_max,
                a_min_lim=a_min_lim,
                a_max_lim=a_max_lim,
            )

        elif binary_representation == "adaptive_range_bce":
            obj = subcls(
                n_qubits,
                a_min=a_min,
                a_max=a_max,
                a_min_lim=a_min_lim,
                a_max_lim=a_max_lim,
                mu=mu,
            )

        obj.binary_representation = binary_representation
        return obj


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
    def __init__(self, n_qubits):
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
    def __init__(self, n_qubits):
        self.n_qubits = n_qubits
        if not ((self.n_qubits % 2) == 0 and self.n_qubits >= 4):
            raise Exception('Number of qubits has to be even and >=4 for real_reduced representation.')

    def evaluate(self, q):
        assert len(q) == self.n_qubits
        m = (self.n_qubits - 2) // 2
        a = -2**m + sum(2**(l - m - 1) * q[l] for l in range(1, 2*m + 2))
        return a

class RealPositive(RealNumber):
    def __init__(self, n_qubits):
        self.n_qubits = n_qubits
        if not ((self.n_qubits % 2) == 1 and self.n_qubits >= 3):
            raise Exception('Number of qubits has to be odd and >=3 for real_positive representation.')

    def evaluate(self, q, a_min=None, a_max=None):
        assert len(q) == self.n_qubits
        m = (self.n_qubits - 1) // 2
        a = sum(2**(l - m) * q[l] for l in range(2*m + 1))
        return a

class Normalized(RealNumber):
    """
    Normalized representation of a real number in the range [0, 1].
    """
    def __init__(self, n_qubits):
        self.n_qubits = n_qubits

    def evaluate(self, q, a_min=None, a_max=None):
        assert len(q) == self.n_qubits
        a = sum(2**l / (2**self.n_qubits - 1) * q[l] for l in range(self.n_qubits))
        return a

class RangeBinary(RealNumber):
    """
    Binary range representation of a real number in the range [a_min, a_max].
    """
    def __init__(self, n_qubits, a_min=None, a_max=None):
        self.n_qubits = n_qubits
        self.a_min = a_min
        self.a_max = a_max
        if self.a_min is None or self.a_max is None:
            raise Exception('a_min and a_max must be set for range representation.')
        if self.a_max <= self.a_min:
            raise Exception('a_max must be greater than a_min.')
        self.weights = [2 ** l for l in range(self.n_qubits)]

    def evaluate(self, q):
        assert len(q) == self.n_qubits

        scale = (self.a_max - self.a_min) / sum(self.weights)
        return self.a_min + scale * sum(self.weights[l] * q[l] for l in range(self.n_qubits))


class RangeBCE(RangeBinary):
    """
    Bounded-coefficient range representation (Karimi & Ronagh, 2019) of a real number in the range [a_min, a_max].
    """
    def __init__(self, n_qubits, a_min=None, a_max=None, mu=None):
        super().__init__(n_qubits, a_min=a_min, a_max=a_max)
        if mu is None:
            raise ValueError('mu must be set for BCE representation.')
        self.mu = mu
        self.weights = self.bounded_coefficient_encoding(self.max_kappa(n_qubits, mu), mu)

    @staticmethod
    def bounded_coefficient_encoding(kappa, mu):
        log2 = lambda n: n.bit_length() - 1

        if kappa < 2 ** (log2(mu) + 1):
            p = log2(kappa)
            return [2 ** i for i in range(p)] + [kappa - (2 ** p - 1)]

        rho = log2(mu) + 1
        nu = kappa - (2 ** rho - 1)
        eta = nu // mu
        w = [2 ** i for i in range(rho)] + [mu] * eta
        if nu - eta * mu != 0:
            w.append(nu - eta * mu)
        return w

    @staticmethod
    def max_kappa(n_qubits, mu):
        rho = mu.bit_length()
        if n_qubits <= rho:
            return 2 ** n_qubits - 1
        return (2 ** rho - 1) + (n_qubits - rho) * mu


class RangeUnary(RangeBCE):
    """
    Unary range representation of a real number in the range [a_min, a_max].
    """
    def __init__(self, n_qubits, a_min=None, a_max=None):
        super().__init__(n_qubits, a_min=a_min, a_max=a_max, mu=1)


class AdaptiveRange:
    """
    Adaptive range representation of a real number in the adaptive range [a_min, a_max].
    """
    def __init__(self, n_qubits, a_min=None, a_max=None, a_min_lim=None, a_max_lim=None):
        super().__init__(n_qubits, a_min, a_max)
        self.a_min_lim = a_min_lim
        self.a_max_lim = a_max_lim

    def set_range(self, a_min, a_max):
        if a_max <= a_min:
            raise Exception('a_max must be greater than a_min.')
        self.a_min = a_min
        self.a_max = a_max

    def update_range(self, a, a_bit_array, a_prev, a_min, a_max, relaxation_factor, verbose=False):

        actions = []
        c = relaxation_factor

        old_min = a_min
        old_max = a_max

        delta = a_max - a_min
        delta_min = a_prev - a_min
        delta_max = a_max - a_prev

        if a < a_prev:
            a_max = a_max - c*delta_max
            if verbose:
                print("  Lowering upper bound.")
            actions.append("max ▼")
        elif a > a_prev:
            a_min = a_min + c*delta_min
            if verbose:
                print("  Raising lower bound.")
            actions.append("min ▲")
        elif a == a_prev:
            a_min = a - delta * (1 + (1 - c)) / 4
            a_max = a + delta * (1 + (1 - c)) / 4
            if verbose:
                print("  Shrinking range.")
            actions.append("shrink ▶◀")

        # Expand range if all qubits at a node are 1 or 0
        if all(b == 1 for b in a_bit_array):
            if verbose:
                print(f"  All qubits are 1, expanding range in positive direction.")
            a_max += 0.25 * delta
            actions.append("max ▲")
        elif all(b == 0 for b in a_bit_array):
            if verbose:
                print(f"  All qubits are 0, expanding range in negative direction.")
            a_min -= 0.25 * delta
            actions.append("min ▼")

        # Enforce limits
        if self.a_min_lim is not None:
            if a_min < self.a_min_lim:
                if verbose:
                    print(f"  Clamping a_min to limit {self.a_min_lim}")
                a_min = self.a_min_lim
                actions.append("min ==")

        if self.a_max_lim is not None:
            if a_max > self.a_max_lim:
                if verbose:
                    print(f"  Clamping a_max to limit {self.a_max_lim}")
                a_max = self.a_max_lim
                actions.append("max ==")

        if old_min != a_min or old_max != a_max:
            if verbose:
                print(f"  Updated range: [{a_min}, {a_max}], ({(a_max - a_min):.6e})")
        else:
            actions.append("none")

        return a_min, a_max, actions

    def re_encode_number(self, a, a_min_new, a_max_new):

        a_bits = self.encode_real_to_bits(a, a_min_new, a_max_new)
        a_new = self.decode_bits_to_real(a_bits, a_min_new, a_max_new)
        rel_err_a_encoded = np.abs(a_new - a) / np.abs(a) if a != 0 else np.nan

        return a_bits, a_new, rel_err_a_encoded

    def encode_real_to_bits(self, value, a_min, a_max):
        if a_max == a_min:
            norm = 0.0
        else:
            norm = (value - a_min) / (a_max - a_min)
        norm = min(max(norm, 0.0), 1.0)
        int_val = int(round(norm * sum(self.weights)))
        bits = [0] * self.n_qubits
        for i in range(self.n_qubits - 1, -1, -1):
            if self.weights[i] <= int_val:
                bits[i] = 1
                int_val -= self.weights[i]
        if int_val != 0:
            raise ValueError("Value not representable with these weights")
        return bits

    def decode_bits_to_real(self, bits, a_min, a_max):
        int_val = sum(w * (b & 1) for w, b in zip(self.weights, bits))
        norm = int_val / sum(self.weights)
        value = a_min + norm * (a_max - a_min)
        return value


class AdaptiveRangeBinary(AdaptiveRange, RangeBinary):
    """Adaptive bounds using the standard binary range encoding."""
    pass


class AdaptiveRangeUnary(AdaptiveRange, RangeUnary):
    """Adaptive bounds using the unary range encoding."""
    pass


class AdaptiveRangeBCE(AdaptiveRange, RangeBCE):
    """Adaptive bounds using the bounded-coefficient range encoding."""
    def __init__(self, n_qubits, a_min=None, a_max=None, a_min_lim=None, a_max_lim=None, mu=None):
        RangeBCE.__init__(self, n_qubits, a_min, a_max, mu)
        self.a_min_lim = a_min_lim
        self.a_max_lim = a_max_lim

RealNumber._registry = {
    "real": Real,
    "real_reduced": RealReduced,
    "real_positive": RealPositive,
    "normalized": Normalized,
    "range_binary": RangeBinary,
    "range_unary": RangeUnary,
    "range_bce": RangeBCE,
    "adaptive_range_binary": AdaptiveRangeBinary,
    "adaptive_range_unary": AdaptiveRangeUnary,
    "adaptive_range_bce": AdaptiveRangeBCE,
}