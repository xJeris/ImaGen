import re


def parse_weighted_prompt(prompt: str) -> str:
    """Convert [text:weight] syntax to compel's (text)weight syntax.

    Examples:
        "[green curtains:1.5] in a room" -> "(green curtains)1.5 in a room"
        "[cat:0.5] on a [red:1.8] couch"  -> "(cat)0.5 on a (red)1.8 couch"
        "plain prompt without weights"     -> "plain prompt without weights"
    """
    pattern = r"\[([^:\]]+):([0-9]*\.?[0-9]+)\]"
    return re.sub(pattern, r"(\1)\2", prompt)
