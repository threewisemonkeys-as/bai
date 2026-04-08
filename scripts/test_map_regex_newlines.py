"""Verify the exact newline pattern before 'Agent the Evoker'."""
import json
import re

with open("logs/dev/mar19/2026-03-21_14-11-06_robust_cot_google_gemini-2.5-flash_mixed_explore/step_1/critical_moments.json") as f:
    moments = json.load(f)

raw_obs = moments[0]["raw_observation"]

direct_start = raw_obs.find("========== Start of Direct Observation ==========")
direct_content = raw_obs[direct_start + len("========== Start of Direct Observation =========="):]

evoker_idx = direct_content.find("Agent the Evoker")
# Show exact bytes before "Agent the Evoker"
preceding = direct_content[evoker_idx-80:evoker_idx]
print("Chars before 'Agent the Evoker':")
for i, c in enumerate(preceding):
    if c == '\n':
        print(f"  [{i}] \\n")
    elif c == ' ':
        pass  # skip spaces for brevity
    else:
        print(f"  [{i}] '{c}'")

# The regex expects \n\n before Agent — but the map has lines with trailing spaces
# So the pattern is: <map line with spaces>\n<blank line with spaces>\n...Agent
# The \n\n requires two consecutive newlines with nothing between them
# But there are spaces (from the 80-char wide screen) filling each "blank" line

# Try a more flexible regex
map_match_flexible = re.search(r"map:\n(.*?)(?=\s*Agent the Evoker)", direct_content, re.DOTALL)
print(f"\nFlexible map regex: {map_match_flexible is not None}")
if map_match_flexible:
    lines = [l for l in map_match_flexible.group(1).split('\n') if l.strip()]
    print(f"  Non-empty map lines: {len(lines)}")

print("\nCONCLUSION: The 'blank' lines between the map and 'Agent the Evoker'")
print("are actually filled with spaces (80-char wide screen), so \\n\\n never matches.")
print("The regex needs \\n\\s*\\n or a more flexible boundary pattern.")
