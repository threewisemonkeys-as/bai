"""Test cursor regex specifically."""
import re

obs = """cursor:
Yourself a ranger
(x=26, y=9)"""

# The perception module's pattern
pattern1 = r"cursor:\n(?:.*?)\(x=(\d+), y=(\d+)\)"
match1 = re.search(pattern1, obs)
print(f"Perception pattern (non-greedy .*?): {match1}")

# The issue: .*? is non-greedy and matches empty string,
# so it tries to match "(x=..." right after "cursor:\n" but finds "Yourself..."
# Let's try with DOTALL
match2 = re.search(pattern1, obs, re.DOTALL)
print(f"With DOTALL: {match2.groups() if match2 else None}")

# The actual text between cursor:\n and (x=...) is "Yourself a ranger\n"
# Without DOTALL, .* doesn't cross newlines, so (?:.*?) matches empty on first line
# then fails because next char is 'Y' not '('
print()
print("Without DOTALL, .*? can't cross the newline between")
print("'Yourself a ranger' and '(x=26, y=9)', so the cursor regex FAILS.")
print("With DOTALL it works fine.")
