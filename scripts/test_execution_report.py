"""Test the new _build_execution_report function."""
import json
import sys
sys.path.insert(0, ".")

from mixed_improve import _build_execution_report

# Load step_1 perception and moments
with open("logs/dev/mar19/2026-03-21_14-11-06_robust_cot_google_gemini-2.5-flash_mixed_explore/step_1/perception.py") as f:
    perception_code = f.read()

with open("logs/dev/mar19/2026-03-21_14-11-06_robust_cot_google_gemini-2.5-flash_mixed_explore/step_1/critical_moments.json") as f:
    moments = json.load(f)

# Build sample_observations from moments
sample_observations = [
    (m["raw_observation"], m["traj_step_number"])
    for m in moments
    if m.get("raw_observation", "").strip()
][:3]

report = _build_execution_report(perception_code, sample_observations)
print("EXECUTION REPORT:")
print(report[:3000])
print()
print(f"Report length: {len(report)} chars")
print(f"Number of samples: {len(sample_observations)}")

# Verify it would catch the degradation
for raw_obs, step_num in sample_observations:
    from mixed_improve import _run_perception_on_observation
    output = _run_perception_on_observation(perception_code, raw_obs)
    line_count = len(output.strip().splitlines()) if output else 0
    has_map = "map:" in raw_obs and len(raw_obs) > 500
    print(f"\nStep {step_num}: output={line_count} lines, has_map={has_map}, degraded={has_map and line_count <= 3}")
