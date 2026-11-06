"""AC power flow in rectangular voltages with Amplify PUBO.
"""

import os
from types import SimpleNamespace

import numpy as np
import pandas as pd
from amplify import Model, Poly, sum_poly
from dimod.views.samples import SampleView
from scipy.optimize import minimize

from engioptiqa.problems.problem import Problem
from engioptiqa.variables.real_number import RealNumber


class PowerFlowData:
    """Read flat case data; bus types: slack=0, PV=1, PQ=2."""

    def __init__(self, path):
        path = os.path.join(path, "")
        self.pd_arr = np.atleast_1d(np.loadtxt(path + "pd.txt"))
        self.qd_arr = np.atleast_1d(np.loadtxt(path + "qd.txt"))
        self.pg_arr = np.atleast_1d(np.loadtxt(path + "pg.txt"))
        self.qg_arr = np.atleast_1d(np.loadtxt(path + "qg.txt"))
        buses = pd.read_csv(path + "bus_type.csv")
        buses["index"] = buses["index"].astype(float).astype(int)
        buses["type"] = buses["type"].astype(float).astype(int)
        self.n_bus = len(buses)
        self.ref_bus = int(buses.loc[buses["type"] == 0, "index"].iloc[0])
        self.pv_buses = buses.loc[buses["type"] == 1, "index"].astype(int).tolist()
        self.pq_buses = buses.loc[buses["type"] == 2, "index"].astype(int).tolist()
        self.g_matrix = self._sparse_to_dense(path + "g.csv", self.n_bus)
        self.b_matrix = self._sparse_to_dense(path + "b.csv", self.n_bus)
        self.v_min = np.loadtxt(path + "vmin.txt")
        self.v_max = np.loadtxt(path + "vmax.txt")
        self.p_min = np.loadtxt(path + "pmin.txt")
        self.p_max = np.loadtxt(path + "pmax.txt")
        self.q_min = np.loadtxt(path + "qmin.txt")
        self.q_max = np.loadtxt(path + "qmax.txt")
        self.cost = np.loadtxt(path + "cost.txt")

    @staticmethod
    def _sparse_to_dense(csv_file, n_bus):
        df = pd.read_csv(csv_file)
        i = df["source"].astype(float).astype(int).to_numpy()
        j = df["target"].astype(float).astype(int).to_numpy()
        w = df["weight"].astype(float).to_numpy()
        matrix = np.zeros((n_bus, n_bus))
        np.add.at(matrix, (i, j), w)
        return matrix


class PowerFlow(Problem):
    """AC P/Q residual with fixed slack voltage 1+0j.

    Active mismatch at PV and PQ buses; reactive mismatch at PQ buses.
    PV-bus voltage setpoints are NOT separately enforced by this formulation.
    """

    def __init__(self, data, output_path=None):
        super().__init__(output_path)
        self.name = "Power Flow"
        self.data = data
        self.print_and_log(self.name + "\n")

    def generate_discretization(
        self,
        n_qubits_per_var=2,
        binary_representation="adaptive_range_binary",
        lower_lim=None,
        upper_lim=None,
        *,
        mu_lower=None,
        mu_upper=None,
        omega_lower=None,
        omega_upper=None,
        mu_center=None,
        omega_center=None,
        delta_mu=None,
        delta_omega=None,
    ):
        """Create binary voltage polynomials for one of two encodings.

        RANGE: 'range_binary' or 'adaptive_range_binary', any positive bits/component.
        By default mu=[0.8, 1.2] and omega=[-0.2, 0.2]. Optional
        component-specific bounds override these defaults. For legacy callers,
        lower_lim/upper_lim (if both supplied) set BOTH component ranges,
        unless explicitly overridden by component-specific bounds.

        INCREMENTAL: 'increment_decrement', exactly two bits/component.
        x_i = center_i + delta_x*(q_plus - q_minus). 00 and 11 leave x_i
        unchanged; 10 increments it; 01 decrements it. Center can be a
        scalar or an n_bus array. Rebuild the model each AQPF iteration.

        The slack bus is always 1+0j in the current physical formulation.
        """
        if not isinstance(n_qubits_per_var, (int, np.integer)) or n_qubits_per_var < 1:
            raise ValueError("n_qubits_per_var must be a positive integer")
        n = self.data.n_bus
        self.initialize_discretization()
        self.n_qubits_per_var = int(n_qubits_per_var)
        self.binary_representation = binary_representation
        mu_polys = [None] * n
        omega_polys = [None] * n

        if binary_representation in ("adaptive_range_binary", "range_binary"):
            if (lower_lim is None) != (upper_lim is None):
                raise ValueError("lower_lim and upper_lim must be provided together")
            common_lower = lower_lim
            common_upper = upper_lim
            ml = mu_lower if mu_lower is not None else (common_lower if common_lower is not None else 0.8)
            mh = mu_upper if mu_upper is not None else (common_upper if common_upper is not None else 1.2)
            wl = omega_lower if omega_lower is not None else (common_lower if common_lower is not None else -0.2)
            wh = omega_upper if omega_upper is not None else (common_upper if common_upper is not None else 0.2)
            ml, mh, wl, wh = map(float, (ml, mh, wl, wh))
            if not all(np.isfinite(v) for v in (ml, mh, wl, wh)) or ml >= mh or wl >= wh:
                raise ValueError("Real/imaginary bounds must be finite and strictly increasing")
            self.mu_range = (ml, mh)
            self.omega_range = (wl, wh)
            self.real_number = RealNumber.create(self.n_qubits_per_var, binary_representation, float(ml), float(mh))
            for i in range(n):
                if i == self.data.ref_bus:
                    mu_polys[i], omega_polys[i] = 1.0, 0.0
                    continue
                q_mu = self.variable_generator.array("Binary", self.n_qubits_per_var)
                q_omega = self.variable_generator.array("Binary", self.n_qubits_per_var)
                # 'range' has no set_range: build two RealNumber instances.
                real_mu = RealNumber.create(self.n_qubits_per_var, binary_representation, float(ml), float(mh))
                real_omega = RealNumber.create(self.n_qubits_per_var, binary_representation, float(wl), float(wh))
                mu_polys[i] = real_mu.evaluate(q_mu)
                omega_polys[i] = real_omega.evaluate(q_omega)

        elif binary_representation in ("increment_decrement", "aqpf_incremental"):
            if self.n_qubits_per_var != 2:
                raise ValueError("Increment/decrement requires exactly two bits per component")
            if delta_mu is None or delta_omega is None:
                raise ValueError("delta_mu and delta_omega are required for increment/decrement")
            if not (np.isfinite(delta_mu) and delta_mu > 0 and np.isfinite(delta_omega) and delta_omega > 0):
                raise ValueError("delta_mu and delta_omega must be positive and finite")
            def center_array(value, default, name):
                if value is None:
                    return np.full(n, default, dtype=float)
                arr = np.asarray(value, dtype=float)
                if arr.ndim == 0:
                    arr = np.full(n, float(arr), dtype=float)
                if arr.shape != (n,) or not np.isfinite(arr).all():
                    raise ValueError(f"{name} must be finite scalar or shape (n_bus,)")
                return arr
            mc = center_array(mu_center, 1.0, "mu_center")
            wc = center_array(omega_center, 0.0, "omega_center")
            self.mu_center = mc.copy()
            self.omega_center = wc.copy()
            self.delta_mu = float(delta_mu)
            self.delta_omega = float(delta_omega)
            for i in range(n):
                if i == self.data.ref_bus:
                    mu_polys[i], omega_polys[i] = 1.0, 0.0
                    continue
                q_mu = self.variable_generator.array("Binary", 2)
                q_omega = self.variable_generator.array("Binary", 2)
                mu_polys[i] = float(mc[i]) + float(delta_mu) * (q_mu[0] - q_mu[1])
                omega_polys[i] = float(wc[i]) + float(delta_omega) * (q_omega[0] - q_omega[1])
        else:
            raise ValueError("PowerFlow supports 'range_binary', 'adaptive_range_binary', or 'increment_decrement'")

        self.mu_polys = mu_polys
        self.omega_polys = omega_polys

    def generate_aqpf_discretization(self, mu_center, omega_center, delta_mu, delta_omega):
        """Backward-compatible delegate; use generate_discretization() in new code."""
        return self.generate_discretization(
            n_qubits_per_var=2,
            binary_representation="increment_decrement",
            mu_center=mu_center,
            omega_center=omega_center,
            delta_mu=delta_mu,
            delta_omega=delta_omega,
        )

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
        if not hasattr(self, "mu_polys"):
            raise RuntimeError("Call generate_discretization() first")
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
        self.print_and_log(f"Number of binary variables: {len(self.binary_model.get_variables())}\n")

    def compute_reference_solution(self, tol=1e-12, max_iter=200):
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
        result = minimize(residual, x0, method="L-BFGS-B", tol=tol,
                          options={"maxiter": max_iter, "gtol": tol})
        if not result.success:
            raise RuntimeError(f"Reference solver did not converge: {result.message}")
        mu = np.ones(n)
        omega = np.zeros(n)
        mu[free] = result.x[:n_free]
        omega[free] = result.x[n_free:]
        self.mu_reference = mu
        self.omega_reference = omega
        self.residual_reference = float(result.fun)
        self.print_and_log(f"Reference residual: {self.residual_reference:.3e}\n")
        return mu, omega

    def compute_residual(self, mu, omega):
        mu = np.asarray(mu, dtype=float)
        omega = np.asarray(omega, dtype=float)
        G, B = self.data.g_matrix, self.data.b_matrix
        P = mu * (G @ mu) + omega * (G @ omega) + omega * (B @ mu) - mu * (B @ omega)
        Q = omega * (G @ mu) - mu * (G @ omega) - mu * (B @ mu) - omega * (B @ omega)
        idx_p = list(self.data.pq_buses) + list(self.data.pv_buses)
        idx_q = list(self.data.pq_buses)
        term_p = np.sum((P[idx_p] + self.data.pd_arr[idx_p] - self.data.pg_arr[idx_p]) ** 2)
        term_q = np.sum((Q[idx_q] + self.data.qd_arr[idx_q] - self.data.qg_arr[idx_q]) ** 2)
        return float((term_p + term_q) / (2 * max(1, self.data.n_bus - 1)))

    def decode_voltages(self, result):
        """Decode Amplify/OpenJij samples or {variable_id: 0/1} dictionaries."""
        def decode_one(poly):
            if isinstance(poly, Poly):
                if isinstance(result, SampleView):
                    return self.decode_amplify_poly_with_bitstring(poly, result._data)
                if isinstance(result, SimpleNamespace):
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
        mu = np.asarray(mu, dtype=float)
        omega = np.asarray(omega, dtype=float)
        return np.hypot(mu, omega), np.rad2deg(np.arctan2(omega, mu))

    @staticmethod
    def angle_error_deg(angle, reference_angle):
        return (np.asarray(angle) - np.asarray(reference_angle) + 180) % 360 - 180

    def analyze_results(self, results=None, result_max=5):
        if results is None and not hasattr(self, "results"):
            raise Exception("analyze_results called without results")
        if results is None:
            results = self.results
        solutions = []
        for i_result, result in enumerate(results):
            bit_array = self.get_bit_array(result)
            mu, omega = self.decode_voltages(result)
            res = self.compute_residual(mu, omega)
            magnitude, angle_deg = self.voltage_observables(mu, omega)
            entry = {
                "bit_array": bit_array, "mu": mu, "omega": omega,
                "voltage_magnitude": magnitude,
                "voltage_angle": np.deg2rad(angle_deg),
                "voltage_angle_deg": angle_deg,
                "residual": res,
                "energy": self.get_energy(i_result),
                "frequency": self.get_frequency(i_result),
            }
            solutions.append(entry)
            if i_result < result_max:
                self._report_solution(i_result, entry)
        self._summarise_best(solutions)
        return solutions

    def _report_solution(self, i_result, entry):
        output = (f"Solution {i_result}\n"
                  f"\tenergy = {entry['energy']:.6g}, frequency = {entry['frequency']}\n"
                  f"\tresidual = {entry['residual']:.3e}\n")
        for i in range(self.data.n_bus):
            output += (f"\tBus {i}: |V| = {entry['voltage_magnitude'][i]:.4f}, "
                       f"theta = {entry['voltage_angle_deg'][i]:+7.3f} deg\n")
        self.print_and_log(output)

    def _summarise_best(self, solutions):
        if not solutions:
            return
        best = min(solutions, key=lambda s: s["residual"])
        output = f"Best annealed residual: {best['residual']:.3e}\n"
        if hasattr(self, "residual_reference"):
            v_ref, theta_ref_deg = self.voltage_observables(self.mu_reference, self.omega_reference)
            err_deg = self.angle_error_deg(best["voltage_angle_deg"], theta_ref_deg)
            output += (f"Reference residual    : {self.residual_reference:.3e}\n"
                       f"|V| max abs error     : {np.max(np.abs(best['voltage_magnitude'] - v_ref)):.3e} p.u.\n"
                       f"theta max abs error   : {np.max(np.abs(err_deg)):.3e} deg\n")
        self.print_and_log(output)