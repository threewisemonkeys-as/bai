#!/usr/bin/env python3
"""Test script to create MiniHack environment with different seeds."""

import gym
import minihack  # noqa: F401 - registers MiniHack envs


def main():
    env = gym.make(
        "MiniHack-Quest-Easy-v0",
        observation_keys=[
            "glyphs",
            "blstats",
            "tty_chars",
            "inv_letters",
            "inv_strs",
            "tty_cursor",
            "tty_colors",
            "screen_descriptions",
        ],
        seeds=list(range(1, 21)),  # Seeds 1 to 20
    )

    print("Testing MiniHack-Quest-Easy-v0 with seeds 1-20\n")
    print("=" * 80)

    for seed in range(1, 21):
        env.seed(seed, seed, reseed=False)
        obs = env.reset(sample_seed=False)  # Returns dict

        print(f"\nSeed: {seed}")
        print(f"  blstats: {obs['blstats'][:5]}...")

        # Print the complete map using tty_chars
        if 'tty_chars' in obs:
            print("  Map:")
            for row in obs['tty_chars']:
                line = "".join(chr(c) if 32 <= c < 127 else "?" for c in row)
                print(f"    {line}")

    print("\n" + "=" * 80)
    print("Done!")
    env.close()


if __name__ == "__main__":
    main()
