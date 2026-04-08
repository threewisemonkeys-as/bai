"""Test cursor regex against actual raw_observation from step_1 moments."""
import json
import re

with open("logs/dev/mar19/2026-03-21_14-11-06_robust_cot_google_gemini-2.5-flash_mixed_explore/step_1/critical_moments.json") as f:
    moments = json.load(f)

raw_obs = moments[0]["raw_observation"]

# The perception module searches the FULL raw_observation (not just the Direct Observation part)
# Check: does cursor regex work on the full raw_observation?
cursor_match_no_dotall = re.search(r"cursor:\n(?:.*?)\(x=(\d+), y=(\d+)\)", raw_obs)
cursor_match_dotall = re.search(r"cursor:\n(?:.*?)\(x=(\d+), y=(\d+)\)", raw_obs, re.DOTALL)

print(f"Cursor match WITHOUT DOTALL: {cursor_match_no_dotall}")
print(f"Cursor match WITH DOTALL: {cursor_match_dotall.groups() if cursor_match_dotall else None}")

# Check map regex with Agent the Evoker
map_match_evoker = re.search(r"map:\n(.*?)(?=\n\nAgent the Evoker)", raw_obs, re.DOTALL)
print(f"\nMap match with 'Agent the Evoker': {map_match_evoker is not None}")
if map_match_evoker:
    lines = [l for l in map_match_evoker.group(1).split('\n') if l.strip()]
    print(f"  Map lines found: {len(lines)}")

# Check inventory regex
inventory_match = re.search(
    r"inventory:\n(.*?)(?=\n\nAgent the Evoker|\n\n========== End of Direct Observation =========)",
    raw_obs, re.DOTALL
)
print(f"\nInventory match: {inventory_match is not None}")
if inventory_match:
    print(f"  First 100 chars: {repr(inventory_match.group(1)[:100])}")

# Now test what the perception module gets as input during scoring
# _run_perception_on_observation passes m.raw_observation directly
# And during rollouts, evaluator.py passes obs["text"]["long_term_context"]
# Are these the same thing?
print("\n" + "="*60)
print("KEY QUESTION: What does raw_observation contain?")
print(f"  Starts with: {repr(raw_obs[:60])}")
print(f"  Contains 'Start of Direct Observation': {'Start of Direct Observation' in raw_obs}")
print(f"  Contains 'inventory:': {'inventory:' in raw_obs}")

# The raw_observation includes BOTH the inventory prefix AND the Direct Observation section
# But during rollouts, perception_fn gets only long_term_context (the Direct Observation part)
# Let's check if the map regex would match the Direct Observation part only
direct_obs_start = raw_obs.find("========== Start of Direct Observation ==========")
if direct_obs_start >= 0:
    direct_obs = raw_obs[direct_obs_start:]
    print(f"\n  Direct Observation section starts at char {direct_obs_start}")

    map_match_direct = re.search(r"map:\n(.*?)(?=\n\nAgent the Evoker)", direct_obs, re.DOTALL)
    print(f"  Map match in Direct Observation only: {map_match_direct is not None}")
