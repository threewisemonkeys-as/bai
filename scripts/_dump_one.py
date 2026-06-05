import asyncio, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import importlib.util
spec = importlib.util.spec_from_file_location("rse", str(Path(__file__).resolve().parent / "resimulate_select_experiment.py"))
rse = importlib.util.module_from_spec(spec); spec.loader.exec_module(rse)
from mixed_improve import _llm_call

OUT = Path("scripts/resim_dumps"); OUT.mkdir(exist_ok=True)
async def main():
    for step in ["009", "012"]:
        prompt, imgs, cands, actual = rse.build_prompt(step)
        resp, cost = await _llm_call(rse.CONFIG, prompt, images=imgs or None)
        f = OUT / f"step_{step}_select_experiment.txt"
        f.write_text(
            f"# STEP {step} | select-among-experiments | {len(imgs)} images attached (not shown)\n"
            f"# score_topk actually chose: {actual!r}\n"
            f"{'='*100}\nPROMPT (text part):\n{'='*100}\n{prompt}\n\n"
            f"{'='*100}\nRESPONSE (cost ${cost:.5f}):\n{'='*100}\n{resp}\n"
        )
        print("wrote", f, "| prompt chars:", len(prompt), "| images:", len(imgs))
asyncio.run(main())
