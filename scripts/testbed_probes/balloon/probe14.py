import os
import sys, json
sys.path.insert(0,"/home/ays57/bai")
from autumn_env import AutumnBenchEnvWrapper
S=os.path.dirname(os.path.abspath(__file__))
env = AutumnBenchEnvWrapper(env_name="balloon", task_type="interactive", data_dir=S+"/data",
                            seed=1, max_episode_steps=100, logging_path=S+"/logs")
r=env.reset(); obs=r[0] if isinstance(r,tuple) else r
def grid(o):
    t=o["text"]["long_term_context"]; return json.loads(t[t.find("[["):])
def gray(g): return sorted((c,r) for r,row in enumerate(g) for c,v in enumerate(row) if v=="gray")
print("init gray:", gray(grid(obs)))
# basket interior at init: cols 6,7,8 rows 13,14 -> click ROW COL = 'click 13 7'
out=env.step("click 13 7"); obs=out[0]
print("after 'click 13 7' -> gray (col,row):", gray(grid(obs)))
print("  EXPECT [(7, 13)] if row/col handled correctly")
out=env.step("noop"); obs=out[0]
print("after noop -> gray:", gray(grid(obs)))
mp=[(c,r) for r,row in enumerate(grid(obs)) for c,v in enumerate(row) if v=="mediumpurple"]
print("  purple rows:", min(p[1] for p in mp), max(p[1] for p in mp), "(balloon rose => rows 4..8)")
