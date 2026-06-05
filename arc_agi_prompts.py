"""ARC-AGI 3 instruction prompts and default knowledge."""


ARC_COLORS = {
    0: "white",
    1: "off-white",
    2: "light-gray",
    3: "gray",
    4: "dark-gray",
    5: "black",
    6: "magenta",
    7: "pink",
    8: "red",
    9: "blue",
    10: "light-blue",
    11: "yellow",
    12: "orange",
    13: "maroon",
    14: "green",
    15: "purple",
}


ARC_ACTION_DESCRIPTIONS = {
    "ACTION1": "ACTION1: Simple action - varies by game (often mapped to up / move-up)",
    "ACTION2": "ACTION2: Simple action - varies by game (often mapped to down / move-down)",
    "ACTION3": "ACTION3: Simple action - varies by game (often mapped to left / move-left)",
    "ACTION4": "ACTION4: Simple action - varies by game (often mapped to right / move-right)",
    "ACTION5": "ACTION5: Simple action - varies by game (often an interaction: select / confirm / rotate / execute)",
    "ACTION6": "ACTION6 x=<int> y=<int>: Complex action - click/select the cell at coordinates x,y (each in 0-63)",
    "ACTION7": "ACTION7: Simple action - varies by game (e.g. a secondary interaction or undo)",
}


def get_arc_instruction_prompt(available_actions: list[str] | None = None) -> str:
    color_desc = "\n".join(f"  {k}: {v}" for k, v in ARC_COLORS.items())

    if available_actions:
        action_lines = [
            ARC_ACTION_DESCRIPTIONS.get(
                name, f"{name}: action available in this game"
            )
            for name in available_actions
        ]
    else:
        action_lines = list(ARC_ACTION_DESCRIPTIONS.values())
    actions_block = "\n".join(action_lines)

    complex_note = (
        '- ACTION6 requires x and y coordinates (e.g. "ACTION6 x=32 y=16").\n'
        if any(line.startswith("ACTION6") for line in action_lines)
        else ""
    )

    return f"""You are playing a game.

The game is played on a 64x64 grid of integer values (0-15), each representing a color:
{color_desc}

Available actions:
<actions>
{actions_block}
</actions>

{complex_note}"""
