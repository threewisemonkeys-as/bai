import arc_agi
from arcengine import GameAction

arc = arc_agi.Arcade()
env = arc.make("ls20", render_mode="terminal")

# See available actions
# print(env.action_space)

# Take an action
# obs = env.step(GameAction.ACTION1)
breakpoint()