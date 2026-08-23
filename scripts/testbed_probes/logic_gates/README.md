Engine probes behind `notes/testbed_investigations/logic_gates.md` (2026-08-23). Run from the
repo root, e.g. `uv run python scripts/testbed_probes/logic_gates/p3_bfs.py`. `lib.py` holds
the interpreter loader (renders after every step, seed >= 1). `p10_variants.py` writes the
NAND/NOR/BUF/XNOR (v1) and two-stage (v2) programs that now live in `autumn_programs/variants/`.
