Engine probes behind `notes/testbed_investigations/balloon.md` (2026-08-23). Run from the repo
root, e.g. `uv run python scripts/testbed_probes/balloon/probe3.py`. `lib.py` holds the
interpreter loader (renders after every step, seed >= 1); `model.py` is the verified
pure-python model of the rules; `validate.py` checks it against the engine. `probe7.py` needs
a `data/programs/balloon.sexp` mirror next to it (it drives `AutumnBenchEnvWrapper` with a
custom `data_dir`).
