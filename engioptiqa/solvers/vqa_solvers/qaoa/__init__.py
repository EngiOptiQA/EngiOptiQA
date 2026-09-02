from .solver_pennylane import QAOASolverPennylane

__all__ = ["QAOASolverPennylane"]

try:
	import qiskit_aqt_provider
except ImportError:
	pass
else:
	from .solver_aqt import QAOASolverAQT

	__all__.append("QAOASolverAQT")
