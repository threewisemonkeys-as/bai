import os
import sys, json
sys.path.insert(0,"/home/ays57/bai")
from autumn_env import AutumnBenchEnvWrapper
env = AutumnBenchEnvWrapper(env_name="balloon", task_type="interactive",
                            data_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"),
                            seed=1, max_episode_steps=100,
                            logging_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs"))
_r = env.reset()
obs = _r[0] if isinstance(_r, tuple) else _r
print("BACKGROUND from interpreter:", env._env.interpreter.get_background())
lt = obs["text"]["long_term_context"]
print("SHORT:", repr(obs["text"]["short_term_context"])[:400])
print("LONG len:", len(lt))
print("LONG head:", lt[:600])
g=json.loads(lt[lt.find("[["):])
from collections import Counter
print("row0:", g[0])
print("colors:", Counter(c for r in g for c in r))
print("actions:", env.get_available_actions()[:8] if hasattr(env,'get_available_actions') else "n/a")
