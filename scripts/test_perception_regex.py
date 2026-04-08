"""Test perception module regexes against actual observation format."""
import re

# Actual long_term_context (direct observation) format from the trajectory
obs = """message:


cursor:
Yourself a ranger
(x=26, y=9)

map:
                        --------------
                        |.@...}......---
                        |.<...}........-
                        |.....}...................>.|
                        |.....}.....r..--------     |
                        |.....}.....r---
                        --------------

Agent the Tenderfoot           St:15 Dx:11 Co:15 In:13 Wi:14 Ch:7 Chaotic S:0
Dlvl:1 $:0 HP:14(14) Pw:2(2) AC:7 Xp:1/0"""

# Test message regex
message_match = re.search(r"message:\n(.*?)(?=\n\ncursor:)", obs, re.DOTALL)
print(f"Message match: {repr(message_match.group(1) if message_match else None)}")

# Test cursor regex
cursor_match = re.search(r"cursor:\n(?:.*?)\(x=(\d+), y=(\d+)\)", obs)
print(f"Cursor match: {cursor_match.groups() if cursor_match else None}")

# Test inventory regex - looks for "Agent the Evoker" which is WRONG for this character class
inventory_match = re.search(
    r"inventory:\n(.*?)(?=\n\nAgent the Evoker|\n\n========== End of Direct Observation =========)",
    obs,
    re.DOTALL,
)
print(f"Inventory match (Evoker pattern): {repr(inventory_match.group(1) if inventory_match else None)}")

# Test stats regex
stats_match = re.search(r"HP:(\d+)\((\d+)\) Pw:(\d+)\((\d+)\) AC:(\d+) Xp:(\d+)/(\d+)", obs)
print(f"Stats match: {stats_match.groups() if stats_match else None}")

# Test map regex - hardcodes "Agent the Evoker"
map_match = re.search(r"map:\n(.*?)(?=\n\nAgent the Evoker)", obs, re.DOTALL)
print(f"Map match (Evoker pattern): {map_match is not None}")

# What about a generic pattern?
map_match2 = re.search(r"map:\n(.*?)(?=\n\nAgent the)", obs, re.DOTALL)
print(f"Map match (generic 'Agent the' pattern): {map_match2 is not None}")
if map_match2:
    print(f"  Map content (first 200 chars): {repr(map_match2.group(1)[:200])}")

print()
print("=" * 60)
print("DIAGNOSIS:")
print("The perception module hardcodes 'Agent the Evoker' in its")
print("regex patterns for both inventory and map parsing.")
print(f"But the actual character class is: 'Agent the Tenderfoot'")
print("This means the map parsing block (lines 110-285) NEVER executes,")
print("so only the basic metadata (message, inventory, stats) is output.")
