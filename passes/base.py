"""
loopfuse/passes/base.py

Pass infrastructure: Pass base class and PassManager.

Design mirrors MLIR's PassManager: passes return a PassResult indicating
whether the program changed, enabling fixed-point iteration. PassManager
runs passes in sequence and collects timing + change statistics.
"""

from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import List, Optional
from ..ir.dialect import AgentProgram


@dataclass
class PassResult:
    pass_name:   str
    changed:     bool
    duration_ms: float
    annotations: List[str] = field(default_factory=list)

    def __repr__(self):
        status = "changed" if self.changed else "no-change"
        return f"PassResult({self.pass_name}, {status}, {self.duration_ms:.2f}ms)"


class Pass:
    """
    Abstract base class for all LoopFuse optimization passes.

    Subclasses implement run(prog) -> PassResult.
    Passes are pure transformers: they receive an AgentProgram, mutate it
    in-place, and return a PassResult describing what changed.
    """

    @property
    def name(self) -> str:
        return self.__class__.__name__

    def run(self, prog: AgentProgram) -> PassResult:
        raise NotImplementedError

    def _result(self, prog: AgentProgram, changed: bool,
                annotations: List[str], start: float) -> PassResult:
        for ann in annotations:
            prog.add_annotation(f"[{self.name}] {ann}")
        return PassResult(
            pass_name   = self.name,
            changed     = changed,
            duration_ms = (time.time() - start) * 1000,
            annotations = annotations,
        )


class PassManager:
    """
    Runs a sequence of passes on an AgentProgram.

    Usage:
        pm = PassManager([KVFusionPass(), SpecPrefillPass(), PhaseSelectPass()])
        results = pm.run(prog)
        for r in results:
            print(r)
    """

    def __init__(self, passes: Optional[List[Pass]] = None):
        self.passes: List[Pass] = passes or []

    def add_pass(self, p: Pass) -> "PassManager":
        self.passes.append(p); return self

    def run(self, prog: AgentProgram) -> List[PassResult]:
        results = []
        for p in self.passes:
            r = p.run(prog)
            results.append(r)
        return results

    def run_until_fixed_point(self, prog: AgentProgram,
                              max_iterations: int = 10) -> List[PassResult]:
        """Re-run passes until no pass changes the program (or max_iterations)."""
        all_results = []
        for _ in range(max_iterations):
            changed = False
            for p in self.passes:
                r = p.run(prog)
                all_results.append(r)
                changed = changed or r.changed
            if not changed:
                break
        return all_results
