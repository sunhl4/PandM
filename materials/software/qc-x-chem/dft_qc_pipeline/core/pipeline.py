"""
Pipeline orchestrator.

The ``Pipeline`` class is the single entry-point for running a
DFT + quantum-embedding calculation.  It:

1. Resolves all pluggable components from the registry.
2. Runs the classical backend (HF/DFT).
3. For each ``FragmentRegion`` in the config:
   a. Builds the embedded Hamiltonian (embedding method).
   b. Maps it to qubits (mapper).
   c. Solves with the quantum solver.
   d. (Optionally) runs a DMET self-consistency loop using 1-RDM feedback.
4. Collects results and returns a ``PipelineResult``.

Benchmark mode: when ``config.benchmark_mode`` is True, every fragment
Hamiltonian is sent to **all** solvers listed in ``config.benchmark_solvers``
and results are compared side-by-side.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from .config import (
    PipelineConfig,
    RegionConfig,
    merge_registry_kwargs,
    validate_pipeline_config,
)
from .interfaces import (
    BackendResult,
    ClassicalBackend,
    EmbeddedHamiltonian,
    EmbeddingMethod,
    PipelineResult,
    QuantumSolver,
    SolverResult,
    SupportsEmbeddedMap,
)
from .registry import registry

logger = logging.getLogger(__name__)


class Pipeline:
    """
    Orchestrates the full DFT + local quantum-embedding computation.

    Parameters
    ----------
    config : PipelineConfig
        Fully-populated configuration object (or loaded from YAML via
        ``PipelineConfig.from_yaml``).
    """

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self._backend: ClassicalBackend | None = None
        self._mapper: SupportsEmbeddedMap | None = None
        self._solver: QuantumSolver | None = None
        self._benchmark_solvers: list[QuantumSolver] = []

    # ------------------------------------------------------------------
    # Component builders (lazy, called inside run())
    # ------------------------------------------------------------------

    def _build_backend(self) -> ClassicalBackend:
        typ, kw = merge_registry_kwargs(self.config.backend)
        return registry.build({"type": typ, **kw}, category="backend")

    def _build_embedding(self) -> EmbeddingMethod:
        typ, kw = merge_registry_kwargs(self.config.embedding)
        return registry.build({"type": typ, **kw}, category="embedding")

    def _build_mapper(self) -> SupportsEmbeddedMap:
        from ..hamiltonian.mappers import build_mapper
        return build_mapper(self.config.mapper)

    def _build_solver(self, solver_cfg=None) -> QuantumSolver:
        cfg_obj = solver_cfg or self.config.solver
        typ, kw = merge_registry_kwargs(cfg_obj)
        return registry.build({"type": typ, **kw}, category="solver")

    # ------------------------------------------------------------------
    # Fragment region helper
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_region(region_cfg: RegionConfig):
        """Import FragmentRegion lazily to avoid circular imports."""
        from ..hamiltonian.fragment_region import FragmentRegion
        return FragmentRegion(
            name=region_cfg.name,
            atom_indices=list(region_cfg.atom_indices),
            orbital_indices=list(region_cfg.orbital_indices),
            nelec=region_cfg.nelec,
            norb=region_cfg.norb,
            localization=region_cfg.localization,
        )

    def _process_region(
        self,
        region_cfg: RegionConfig,
        backend_result: BackendResult,
        mapper: SupportsEmbeddedMap,
        solver: QuantumSolver,
        benchmark_solvers: list[tuple[str, QuantumSolver]],
    ) -> tuple[str, SolverResult | None, EmbeddedHamiltonian | None, float | None]:
        """
        Run embed → map → solve (and DMET loop) for one region.

        Each call uses a **fresh** embedding instance (L1). When ``parallel_regions``
        is used, pass a **dedicated** ``solver`` (and benchmark solvers) per call
        so threads do not share mutable solver state.

        Returns the last ``EmbeddedHamiltonian`` from the embedding loop (for
        diagnostics such as DMET ``μ`` in ``extra``).
        """
        region = self._resolve_region(region_cfg)
        embedding = self._build_embedding()
        logger.info(
            "Processing region=%r solver=%s embedding=%s",
            region.name,
            self.config.solver.type,
            self.config.embedding.type,
        )

        emb_H: EmbeddedHamiltonian | None = None
        sol: SolverResult | None = None
        mapped_last: tuple | None = None
        converged = False
        iteration = 0
        max_iter = self.config.embedding.max_iter

        while not converged and iteration < max_iter:
            iteration += 1
            logger.debug(
                "region=%r embedding_iteration=%d/%d",
                region.name,
                iteration,
                max_iter,
            )

            emb_H = embedding.embed(backend_result, region)
            mapped_last = mapper.map(emb_H)
            H_qubit, n_particles, n_orbs = mapped_last

            sol = solver.solve(H_qubit, n_particles, n_orbs)
            logger.debug(
                "region=%r solver=%s fragment_energy=%.10f Ha",
                region.name,
                self.config.solver.type,
                sol.energy,
            )

            if sol.rdm1 is not None:
                converged = embedding.update_from_rdm(sol.rdm1)
            else:
                converged = True

        if sol is None:
            return region.name, None, None, None

        if self.config.embedding.type == "dmet":
            dmet_mudn = float(getattr(embedding, "_last_mu_times_deltaN_ha", 0.0))
        else:
            dmet_mudn = None

        if self.config.benchmark_mode and emb_H is not None and mapped_last is not None:
            bH_qubit, bn_particles, bn_orbs = mapped_last
            bench_region_results = sol.extra.setdefault("benchmark", {})
            for bname, bsolver in benchmark_solvers:
                bsol = bsolver.solve(bH_qubit, bn_particles, bn_orbs)
                bench_region_results[bname] = {
                    "energy": bsol.energy,
                    "converged": bsol.converged,
                }
                logger.info(
                    "region=%r [benchmark] solver=%s energy=%.10f Ha",
                    region.name,
                    bname,
                    bsol.energy,
                )

        return region.name, sol, emb_H, dmet_mudn

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(self) -> PipelineResult:
        """Execute the full pipeline and return aggregated results."""
        validate_pipeline_config(self.config)
        # --- Step 1: build components ---
        logger.info(
            "Building pipeline components (solver=%s, embedding=%s)",
            self.config.solver.type,
            self.config.embedding.type,
        )
        backend = self._build_backend()
        mapper = self._build_mapper()
        solver = self._build_solver()

        benchmark_solvers: list[tuple[str, QuantumSolver]] = []
        if self.config.benchmark_mode:
            from .config import SolverConfig
            for sname in self.config.benchmark_solvers:
                bench_cfg = SolverConfig(type=sname)
                benchmark_solvers.append((sname, self._build_solver(bench_cfg)))

        # --- Step 2: classical backend ---
        bc = self.config.backend
        logger.info("Running %s backend on geometry: %s", bc.type, bc.geometry[:40])
        backend_result: BackendResult = backend.run(
            geometry=bc.geometry,
            basis=bc.basis,
            method=bc.method,
            xc=bc.xc,
            charge=bc.charge,
            spin=bc.spin,
            density_fit=bc.density_fit,
            auxbasis=bc.auxbasis,
        )
        logger.info("Backend HF/DFT energy: %.10f Ha", backend_result.energy_hf)

        # --- Step 3: per-fragment loop (optional parallel threads for independent regions) ---
        fragment_results: dict[str, SolverResult] = {}
        dmet_mu_by_region: dict[str, float] = {}
        dmet_mu_times_deltaN_by_region: dict[str, float] = {}

        use_parallel = (
            self.config.parallel_regions
            and len(self.config.regions) > 1
            and not self.config.benchmark_mode
        )
        if use_parallel:
            n = len(self.config.regions)
            max_workers = self.config.max_parallel_workers or min(32, n)
            logger.info(
                "Running %d regions in parallel (max_workers=%d)",
                n,
                max_workers,
            )
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                futures = []
                for region_cfg in self.config.regions:
                    # Dedicated solvers per region — avoid shared mutable VQE/SQD state across threads.
                    reg_solver = self._build_solver()
                    futures.append(
                        ex.submit(
                            self._process_region,
                            region_cfg,
                            backend_result,
                            mapper,
                            reg_solver,
                            [],
                        )
                    )
                for fut in as_completed(futures):
                    name, sol, emb_H, dmet_mudn = fut.result()
                    if sol is not None:
                        fragment_results[name] = sol
                        if (
                            emb_H is not None
                            and self.config.embedding.type == "dmet"
                            and emb_H.extra
                        ):
                            mu = emb_H.extra.get("mu")
                            if mu is not None:
                                dmet_mu_by_region[name] = float(mu)
                        if dmet_mudn is not None:
                            dmet_mu_times_deltaN_by_region[name] = float(dmet_mudn)
        else:
            if self.config.parallel_regions and self.config.benchmark_mode:
                logger.info(
                    "parallel_regions ignored while benchmark_mode is True; "
                    "running regions sequentially."
                )
            for region_cfg in self.config.regions:
                name, sol, emb_H, dmet_mudn = self._process_region(
                    region_cfg,
                    backend_result,
                    mapper,
                    solver,
                    benchmark_solvers,
                )
                if sol is not None:
                    fragment_results[name] = sol
                    if (
                        emb_H is not None
                        and self.config.embedding.type == "dmet"
                        and emb_H.extra
                    ):
                        mu = emb_H.extra.get("mu")
                        if mu is not None:
                            dmet_mu_by_region[name] = float(mu)
                    if dmet_mudn is not None:
                        dmet_mu_times_deltaN_by_region[name] = float(dmet_mudn)

        # total_energy: sum of fragment solver energies when at least one region succeeded.
        # If no fragment completed (e.g. empty regions), fall back to backend HF/DFT reference.
        if fragment_results:
            total_energy = sum(r.energy for r in fragment_results.values())
        else:
            total_energy = backend_result.energy_hf

        logger.info("Pipeline finished. Total fragment energy: %.10f Ha", total_energy)

        e_back = float(backend_result.energy_hf)
        sum_frag = (
            float(sum(r.energy for r in fragment_results.values()))
            if fragment_results
            else None
        )
        delta_hf_frag = (
            (e_back - sum_frag) if sum_frag is not None else None
        )
        frag_by_name = (
            {k: float(v.energy) for k, v in fragment_results.items()}
            if fragment_results
            else {}
        )

        dmet_corr_sum: float | None = None
        if self.config.embedding.type == "dmet" and dmet_mu_times_deltaN_by_region:
            dmet_corr_sum = float(sum(dmet_mu_times_deltaN_by_region.values()))

        inter_pc: float | None = None
        if (
            self.config.embedding.type == "dmet"
            and self.config.include_inter_fragment_point_charge
            and len(self.config.regions) > 1
        ):
            from ..postprocessing.inter_fragment_estimate import (
                inter_fragment_point_charge_from_backend,
            )

            inter_pc = inter_fragment_point_charge_from_backend(
                backend_result, self.config.regions
            )

        energy_corrections: dict[str, object] = {
            "dmet_inter_fragment_ha": inter_pc,
            "dmet_inter_fragment_model": (
                "mulliken_point_charge_between_region_atom_groups"
                if inter_pc is not None
                else None
            ),
            "dmet_correlation_potential_ha": dmet_corr_sum,
            "dmet_correlation_potential_model": (
                "sum_over_regions_mu_times_deltaN_electrons"
                if dmet_corr_sum is not None
                else None
            ),
            "notes": (
                "dmet_correlation_potential_ha is Σ_A (μ·ΔN)_A at the last DMET update "
                "(Ha); ΔN = N_MF − N_solver on the fragment block — vanishes when "
                "self-consistency is reached. dmet_inter_fragment_ha (when set) is a "
                "classical Mulliken q_i q_j / R_ij sum **between** region atom groups "
                "(not the full quantum DMET bath coupling). Other fields remain diagnostics "
                "vs backend_reference / sum_fragment_energies."
            ),
            "backend_reference_energy_ha": e_back,
            "sum_fragment_energies_ha": sum_frag,
            "delta_backend_minus_fragments_ha": delta_hf_frag,
            "fragment_energies_by_region_ha": frag_by_name,
            "dmet_mu_by_region_ha": dmet_mu_by_region or None,
            "dmet_mu_times_deltaN_by_region_ha": dmet_mu_times_deltaN_by_region or None,
        }

        return PipelineResult(
            total_energy=total_energy,
            fragment_results=fragment_results,
            backend_result=backend_result,
            extra={
                "total_energy_note": (
                    "When fragment_results is non-empty, total_energy is the sum of fragment "
                    "solver energies (inter-fragment coupling and full-system DMET "
                    "correlation-energy corrections are not included). When no fragment "
                    "completed, total_energy equals backend_result.energy_hf as a reference "
                    "only; do not mix these interpretations without checking fragment_results."
                ),
                "energy_corrections": energy_corrections,
            },
        )
