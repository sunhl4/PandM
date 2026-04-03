"""
quantum_chem_bench.quantum_solvers — quantum algorithm solvers.

Importing this package registers:
  vqe_uccsd, vqe_hea, vqe_kupccgsd, adapt_vqe, qpe, qpe_full, sqd, qse
"""

from quantum_chem_bench.quantum_solvers import (  # noqa: F401
    vqe_solver,
    adapt_vqe_solver,
    qpe_solver,
    sqd_solver,
    qse_solver,
)
