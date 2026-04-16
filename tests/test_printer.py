"""tests/test_printer.py — unit tests for IRPrinter."""
import pytest
from loopfuse.ir.printer import IRPrinter
from loopfuse.passes import PassManager, SpecPrefillPass, PhaseSelectPass


class TestIRPrinter:
    def test_print_returns_string(self, gpt2_3step):
        p = IRPrinter()
        out = p.print(gpt2_3step)
        assert isinstance(out, str)
        assert len(out) > 0

    def test_program_name_in_output(self, gpt2_3step):
        out = IRPrinter().print(gpt2_3step)
        assert gpt2_3step.name in out

    def test_step_markers_in_output(self, gpt2_3step):
        out = IRPrinter().print(gpt2_3step)
        for i in range(3):
            assert f"agent.step[{i}]" in out

    def test_phase_tags_in_output(self, gpt2_3step):
        out = IRPrinter().print(gpt2_3step)
        assert "prefill" in out or "decode" in out
        assert "io" in out

    def test_compiler_star_after_passes(self, gpt2_3step):
        PassManager([SpecPrefillPass()]).run(gpt2_3step)
        out = IRPrinter().print(gpt2_3step)
        assert "★" in out  # compiler-inserted marker

    def test_pass_annotations_in_output(self, gpt2_3step):
        PassManager([SpecPrefillPass(), PhaseSelectPass()]).run(gpt2_3step)
        out = IRPrinter().print(gpt2_3step)
        assert "Pass annotations" in out

    def test_summary_table_returns_string(self, gpt2_3step):
        table = IRPrinter().print_summary_table(gpt2_3step)
        assert isinstance(table, str)
        assert "prefill" in table or "decode" in table

    def test_color_mode_toggle(self, gpt2_3step):
        plain  = IRPrinter(use_color=False).print(gpt2_3step)
        color  = IRPrinter(use_color=True).print(gpt2_3step)
        # Color mode adds ANSI escape codes; plain does not
        assert "\033[" not in plain
        assert len(color) > len(plain)  # ANSI codes add bytes
