"""Debug why map regex fails even with 'Agent the Evoker' present."""
import json
import re

with open("logs/dev/mar19/2026-03-21_14-11-06_robust_cot_google_gemini-2.5-flash_mixed_explore/step_1/critical_moments.json") as f:
    moments = json.load(f)

raw_obs = moments[0]["raw_observation"]

# Check: is "Agent the Evoker" actually in the raw observation?
print(f"'Agent the Evoker' in raw_obs: {'Agent the Evoker' in raw_obs}")

# Find the exact context around "Agent the Evoker"
idx = raw_obs.find("Agent the Evoker")
if idx >= 0:
    print(f"Found at index {idx}")
    print(f"Context: {repr(raw_obs[idx-20:idx+60])}")

# The regex is: map:\n(.*?)(?=\n\nAgent the Evoker)
# This requires TWO newlines before "Agent the Evoker"
# Let's check what's actually between the map and the Agent line
map_idx = raw_obs.rfind("map:")
if map_idx >= 0:
    between = raw_obs[map_idx:idx]
    # Count trailing newlines before "Agent the Evoker"
    end_of_map_content = between.rstrip()
    trailing = between[len(end_of_map_content):]
    print(f"\nBetween 'map:' and 'Agent the Evoker':")
    print(f"  Trailing whitespace before Agent: {repr(trailing)}")
    print(f"  Number of newlines: {trailing.count(chr(10))}")

# Also check: the raw_observation has inventory BEFORE Direct Observation
# So the map regex would find the map inside Direct Observation section
# But the "map:\n" pattern might also match something in inventory description?

# What about the actual evaluator.py - what does it pass?
# Line 324: perception_fn(long_term_text) where long_term_text = obs["text"]["long_term_context"]
# This is JUST the direct observation text, not the full obs with inventory prefix

# But in score_perception_on_moments, it passes m.raw_observation
# Let's check what raw_observation is in the moment extraction code
print("\n" + "="*60)
print("The raw_observation field contains BOTH inventory + Direct Observation")
print("But during actual rollouts, perception gets only long_term_context")
print("(the content inside the Direct Observation delimiters)")
print()
print("Let's test the perception on JUST the Direct Observation part:")

direct_start = raw_obs.find("========== Start of Direct Observation ==========")
direct_end = raw_obs.find("========== End of Direct Observation ==========")
if direct_start >= 0 and direct_end >= 0:
    # The long_term_context is the content between delimiters
    direct_content = raw_obs[direct_start + len("========== Start of Direct Observation =========="):direct_end]
    print(f"Direct obs content length: {len(direct_content)}")

    map_match = re.search(r"map:\n(.*?)(?=\n\nAgent the Evoker)", direct_content, re.DOTALL)
    print(f"Map match on direct content: {map_match is not None}")

    # Check what's actually around Agent the Evoker in direct content
    evoker_idx = direct_content.find("Agent the Evoker")
    if evoker_idx >= 0:
        print(f"Context before Evoker: {repr(direct_content[evoker_idx-30:evoker_idx+30])}")
