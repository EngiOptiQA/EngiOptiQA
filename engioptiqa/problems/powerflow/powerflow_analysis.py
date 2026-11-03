"""AC power flow in rectangular voltages, with one-shot and AQPF encodings.

V_i = mu_i + 1j * omega_i. The physical P/Q mismatch objective is shared by
both workflows. The one-shot workflow encodes absolute voltages with RealNumber;
the iterative AQPF workflow encodes local signed voltage increments.
"""
from amplify import Model, Poly, sum_poly
from dimod.views.samples import SampleView
import numpy as np
import os
import pandas as pd
from scipy.optimize import minimize
from types import SimpleNamespace

from engioptiqa.problems.problem import Problem
from engioptiqa.variables.real_number import RealNumber


class PowerFlowData:
    """Read the existing flat-case data layout (slack=0, PV=1, PQ=2)."""

    def __init__(self, path):
        path = os.path.join(path, '')
        self.pd_arr = np.atleast_1d(np.loadtxt(path + 'pd.txt'))
        self.qd_arr = np.atleast_1d(np.loadtxt(path + 'qd.txt'))
        self.pg_arr = np.atleast_1d(np.loadtxt(path + 'pg.txt'))
        self.qg_arr = np.atleast_1d(np.loadtxt(path + 'qg.txt'))
        buses = pd.read_csv(path + 'bus_type.csv')
        buses['index'] = buses['index'].astype(float).astype(int)
        buses['type'] = buses['type'].astype(float).astype(int)
        self.n_bus = len(buses)
        self.ref_bus = int(buses.loc[buses['type'] == 0, 'index'].iloc[0])
        self.pv_buses = buses.loc[buses['type'] == 1, 'index'].astype(int).tolist()
        self.pq_buses = buses.loc[buses['type'] == 2, 'index'].astype(int).tolist()
        self.g_matrix = self._sparse_to_dense(path + 'g.csv', self.n_bus)
        self.b_matrix = self._sparse_to_dense(path + 'b.csv', self.n_bus)
        self.v_min = np.loadtxt(path + 'vmin.txt')
        self.v_max = np.loadtxt(path + 'vmax.txt')
        self.p_min = np.loadtxt(path + 'pmin.txt')
        self.p_max = np.loadtxt(path + 'pmax.txt')
        self.q_min = np.loadtxt(path + 'qmin.txt')
        self.q_max = np.loadtxt(path + 'qmax.txt')
        self.cost = np.loadtxt(path + 'cost.txt')

    @staticmethod
    def _sparse_to_dense(csv_file, n_bus):
        df = pd.read_csv(csv_file)
        i = df['source'].astype(float).astype(int).to_numpy()
        j = df['target'].astype(float).astype(int).to_numpy()
        w = df['weight'].astype(float).to_numpy()
        M = np.zeros((n_bus, n_bus))
        np.add.at(M, (i, j), w)
        return M


class PowerFlow(Problem):
    """AC P/Q residual encoded as an Amplify polynomial.

    For non-slack buses: active mismatch at PQ and PV buses, reactive mismatch
    at PQ buses. The slack voltage is pinned to 1 + 0j. This does not enforce
    PV voltage setpoints; that limitation is inherited from the existing model.
    """

    def __init__(self, data, output_path=None):
        super().__init__(output_path)
        self.name = 'Power Flow'
        self.data = data
        self.print_and_log(self.name + '\n')

    def generate_discretization(self, n_qubits_per_var, binary_representation,
                                lower_lim=None, upper_lim=None):
        """Original EngiOptiQA one-shot encoding (absolute voltages)."""
        self.initialize_discretization()
        self.n_qubits_per_var = n_qubits_per_var
        self.binary_representation = binary_representation
        self.real_number = RealNumber(
            n_qubits_per_var, binary_representation, lower_lim, upper_lim
        )
        mu_polys = [None] * self.data.n_bus
        omega_polys = [None] * self.data.n_bus
        for i in range(self.data.n_bus):
            if i == self.data.ref_bus:
                mu_polys[i] = 1.0
                omega_polys[i] = 0.0
                continue
            q_mu = self.variable_generator.array('Binary', n_qubits_per_var)
            q_omega = self.variable_generator.array('Binary', n_qubits_per_var)
            if binary_representation == 'adaptive_range':
                self.real_number.set_range(lower_lim, upper_lim)
            # mu_polys[i] = self.real_number.evaluate(q_mu)
            # omega_polys[i] = self.real_number.evaluate(q_omega)
            # Rectangular voltage components
            mu_lower, mu_upper = 0.8, 1.2
            omega_lower, omega_upper = -0.2, 0.2
            self.real_number.set_range(mu_lower, mu_upper)
            mu_polys[i] = self.real_number.evaluate(q_mu)
            self.real_number.set_range(omega_lower, omega_upper)
            omega_polys[i] = self.real_number.evaluate(q_omega)

        self.mu_polys = mu_polys
        self.omega_polys = omega_polys

    def generate_aqpf_discretization(self, mu_center, omega_center,
                                     delta_mu, delta_omega):
        """Encode AQPF's local voltage updates (two bits per component).

        mu_i = mu_center[i] + delta_mu * (x_mu_plus - x_mu_minus)
        omega_i = omega_center[i] + delta_omega * (x_omega_plus - x_omega_minus)

        Each iteration rebuilds the polynomial around its current center.
        """
        n = self.data.n_bus
        mu_center = np.asarray(mu_center, dtype=float)
        omega_center = np.asarray(omega_center, dtype=float)
        if mu_center.shape != (n,) or omega_center.shape != (n,):
            raise ValueError('AQPF center arrays must each have shape (n_bus,)')
        if (not np.all(np.isfinite(mu_center)) or
                not np.all(np.isfinite(omega_center))):
            raise ValueError('AQPF centers must be finite')
        if not (np.isfinite(delta_mu) and delta_mu > 0 and
                np.isfinite(delta_omega) and delta_omega > 0):
            raise ValueError('AQPF increments must be positive and finite')
        self.initialize_discretization()
        self.n_qubits_per_var = 2
        self.binary_representation = 'aqpf_incremental'
        self.mu_polys = []
        self.omega_polys = []
        for i in range(n):
            if i == self.data.ref_bus:
                self.mu_polys.append(1.0)
                self.omega_polys.append(0.0)
                continue
            q_mu = self.variable_generator.array('Binary', 2)
            q_omega = self.variable_generator.array('Binary', 2)
            self.mu_polys.append(float(mu_center[i]) + float(delta_mu) *
                                 (q_mu[0] - q_mu[1]))
            self.omega_polys.append(float(omega_center[i]) + float(delta_omega) *
                                    (q_omega[0] - q_omega[1]))

    def _injection_polys(self, mu, omega):
        n = self.data.n_bus
        G, B = self.data.g_matrix, self.data.b_matrix
        p_poly, q_poly = [], []
        for i in range(n):
            p_poly.append(sum_poly(
                mu[i] * G[i, j] * mu[j] + omega[i] * G[i, j] * omega[j]
                + omega[i] * B[i, j] * mu[j] - mu[i] * B[i, j] * omega[j]
                for j in range(n)
            ))
            q_poly.append(sum_poly(
                omega[i] * G[i, j] * mu[j] - mu[i] * G[i, j] * omega[j]
                - mu[i] * B[i, j] * mu[j] - omega[i] * B[i, j] * omega[j]
                for j in range(n)
            ))
        return p_poly, q_poly

    def generate_problem_formulation(self):
        if not hasattr(self, 'mu_polys'):
            raise RuntimeError('Call generate_discretization() or '
                               'generate_aqpf_discretization() first.')
        data = self.data
        p_i, q_i = self._injection_polys(self.mu_polys, self.omega_polys)
        active = sum_poly(
            (p_i[i] + data.pd_arr[i] - data.pg_arr[i]) ** 2
            for i in (data.pq_buses + data.pv_buses)
        )
        reactive = sum_poly(
            (q_i[i] + data.qd_arr[i] - data.qg_arr[i]) ** 2
            for i in data.pq_buses
        )
        n_eq = max(1, data.n_bus - 1)
        self.poly = (active + reactive) / (2 * n_eq)
        self.binary_model = Model(self.poly)
        self.print_and_log(
            f'Number of binary variables: {len(self.binary_model.get_variables())}\n'
        )

    def compute_reference_solution(self, tol=1e-12, max_iter=200):
        """Continuous minimizer used only as a validation reference."""
        n = self.data.n_bus
        free = [i for i in range(n) if i != self.data.ref_bus]
        n_free = len(free)

        def residual(x):
            mu = np.ones(n)
            omega = np.zeros(n)
            mu[free] = x[:n_free]
            omega[free] = x[n_free:]
            return self.compute_residual(mu, omega)

        x0 = np.concatenate([np.ones(n_free), np.zeros(n_free)])
        result = minimize(residual, x0, method='L-BFGS-B', tol=tol,
                          options={'maxiter': max_iter, 'gtol': tol})
        if not result.success:
            raise RuntimeError(f'Reference solver did not converge: {result.message}')
        mu = np.ones(n)
        omega = np.zeros(n)
        mu[free] = result.x[:n_free]
        omega[free] = result.x[n_free:]
        self.mu_reference = mu
        self.omega_reference = omega
        self.residual_reference = float(result.fun)
        self.print_and_log(f'Reference residual: {self.residual_reference:.3e}\n')
        return mu, omega

    def compute_residual(self, mu, omega):
        """Evaluate the original physical residual (not quadratized energy)."""
        mu = np.asarray(mu, dtype=float)
        omega = np.asarray(omega, dtype=float)
        G = self.data.g_matrix
        B = self.data.b_matrix
        P = mu * (G @ mu) + omega * (G @ omega) \
            + omega * (B @ mu) - mu * (B @ omega)
        Q = omega * (G @ mu) - mu * (G @ omega) \
            - mu * (B @ mu) - omega * (B @ omega)
        idx_p = list(self.data.pq_buses) + list(self.data.pv_buses)
        idx_q = list(self.data.pq_buses)
        term_p = np.sum((P[idx_p] + self.data.pd_arr[idx_p]
                         - self.data.pg_arr[idx_p]) ** 2)
        term_q = np.sum((Q[idx_q] + self.data.qd_arr[idx_q]
                         - self.data.qg_arr[idx_q]) ** 2)
        return float((term_p + term_q) / (2 * max(1, self.data.n_bus - 1)))

    def decode_voltages(self, result):
        """Decode voltages using the original EngiOptiQA result interface."""
        def decode_one(poly):
            if isinstance(poly, Poly):
                if type(result) is SampleView:
                    return self.decode_amplify_poly_with_bitstring(poly, result._data)
                if type(result) is SimpleNamespace:
                    return self.decode_amplify_poly_with_bitstring(poly, result.values)
                if isinstance(result, dict):
                    return self.decode_amplify_poly_with_bitstring(poly, result)
                return poly.decode(result.values)
            return float(poly)
        mu = np.array([decode_one(p) for p in self.mu_polys], dtype=float)
        omega = np.array([decode_one(p) for p in self.omega_polys], dtype=float)
        return mu, omega

    @staticmethod
    def voltage_observables(mu, omega):
        """Magnitudes in p.u. and wrapped phase angles in degrees."""
        mu = np.asarray(mu, dtype=float)
        omega = np.asarray(omega, dtype=float)
        return np.hypot(mu, omega), np.rad2deg(np.arctan2(omega, mu))

    @staticmethod
    def angle_error_deg(angle, reference_angle):
        """Wrap angular differences into [-180, 180] degrees."""
        return (np.asarray(angle) - np.asarray(reference_angle) + 180) % 360 - 180

    def analyze_results(self, results=None, result_max=5):
        if results is None and not hasattr(self, 'results'):
            raise Exception('analyze_results called without results to analyse.')
        if results is None:
            results = self.results
        solutions = []
        for i_result, result in enumerate(results):
            bit_array = self.get_bit_array(result)
            mu, omega = self.decode_voltages(result)
            res = self.compute_residual(mu, omega)
            magnitude, angle_deg = self.voltage_observables(mu, omega)
            entry = {
                'bit_array': bit_array,
                'mu': mu,
                'omega': omega,
                'voltage_magnitude': magnitude,
                'voltage_angle': np.deg2rad(angle_deg),  # keep legacy radians
                'voltage_angle_deg': angle_deg,
                'residual': res,
                'energy': self.get_energy(i_result),
                'frequency': self.get_frequency(i_result),
            }
            solutions.append(entry)
            if i_result < result_max:
                self._report_solution(i_result, entry)
        self._summarise_best(solutions)
        return solutions

    def _report_solution(self, i_result, entry):
        output = (
            f'Solution {i_result}\n'
            f"\tenergy = {entry['energy']:.6g}, "
            f"frequency = {entry['frequency']}\n"
            f"\tresidual = {entry['residual']:.3e}\n"
        )
        for i in range(self.data.n_bus):
            output += (
                f"\tBus {i}: |V| = {entry['voltage_magnitude'][i]:.4f}, "
                f"theta = {entry['voltage_angle_deg'][i]:+7.3f} deg\n"
            )
        self.print_and_log(output)

    def _summarise_best(self, solutions):
        if not solutions:
            return
        best = min(solutions, key=lambda s: s['residual'])
        output = f"Best annealed residual: {best['residual']:.3e}\n"
        if hasattr(self, 'residual_reference'):
            v_ref, theta_ref_deg = self.voltage_observables(
                self.mu_reference, self.omega_reference)
            err_deg = self.angle_error_deg(best['voltage_angle_deg'], theta_ref_deg)
            output += (
                f'Reference residual    : {self.residual_reference:.3e}\n'
                f"|V| max abs error     : {np.max(np.abs(best['voltage_magnitude'] - v_ref)):.3e} p.u.\n"
                f'theta max abs error   : {np.max(np.abs(err_deg)):.3e} deg\n'
            )
        self.print_and_log(output)
