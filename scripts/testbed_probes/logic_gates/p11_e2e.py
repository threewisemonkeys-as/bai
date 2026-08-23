import sys, json
sys.path.insert(0,"/home/ays57/bai")
from autumn_env import AutumnBenchEnvWrapper
import os; D=os.path.dirname(os.path.abspath(__file__))
env = AutumnBenchEnvWrapper("logic_gates", task_type="interactive", seed=1,
                            data_dir=f"{D}/bench", logging_path=f"{D}/logs")
obs,_ = env.reset(seed=1)
txt = obs["text"]["long_term_context"]
print("obs chars:", len(txt))
print("first 200:", txt[:200].replace("\n"," | "))
acts = env.language_action_space()
print("n actions:", len(acts), "| sample:", acts[:6], "...", acts[-3:])
