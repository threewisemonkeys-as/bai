"""Run the step_1 perception module on a step_1 moment's raw_observation."""
import json
import sys
sys.path.insert(0, ".")

from mixed_improve import _run_perception_on_observation

# Load step_1 perception
with open("logs/dev/mar19/2026-03-21_14-11-06_robust_cot_google_gemini-2.5-flash_mixed_explore/step_1/perception.py") as f:
    perception_code = f.read()

# Load step_1 moments
with open("logs/dev/mar19/2026-03-21_14-11-06_robust_cot_google_gemini-2.5-flash_mixed_explore/step_1/critical_moments.json") as f:
    moments = json.load(f)

raw_obs = moments[0]["raw_observation"]

# Run perception
output = _run_perception_on_observation(perception_code, raw_obs)
print("PERCEPTION OUTPUT ON STEP_1 MOMENT:")
print(output)
print()
print(f"Output lines: {len(output.splitlines())}")
print()
print("=" * 60)
print("ANALYSIS:")
if "Hazard" in output or "CRITICAL" in output or "position" in output:
    print("Map parsing IS producing output (hazards, positions, etc.)")
else:
    print("Map parsing is NOT producing output - only basic metadata")
    print("This means the perception was ALREADY broken during step_1 scoring")
    print("The 0.6 moment_score was achieved with only basic metadata output")
