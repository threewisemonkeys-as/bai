import gymnasium as gym
import minihack
import matplotlib.pyplot as plt
from nle import nethack

MOVE_ACTIONS = tuple(nethack.CompassCardinalDirection)
env = gym.make(
   "MiniHack-Eat-v0",
   observation_keys=("glyphs", "chars", "colors", "pixel"),
   actions=MOVE_ACTIONS,
)


obs, info = env.reset()

plt.figure(figsize=(10, 8))
plt.axis('off')
plt.tight_layout()
plt.imshow(obs['pixel'])
plt.show()

print(f"Action space: {env.action_space}")


obs, reward, done, truncated, info = env.step(2)
plt.imshow(obs['pixel'])
plt.show()

