"""wordle_ui.py — Shared UI utilities for parsing and formatting response patterns.

Used by both the interactive game (wordle.py) and the swarm CLI (erd_search.py).
"""


def _is_gray_char(ch: str) -> bool:
    """Accept 0, _, and any non-alphanumeric character as gray.

    Accommodates mobile keyboards where -- becomes an em dash
    and .. becomes a period-space.
    """
    return ch == '0' or ch == '_' or not ch.isalnum()


def parse_pattern(s: str) -> int:
    """Parse a 5-character response string to its integer code.

    g = green, y = yellow, 0 / _ / any non-alphanumeric = gray.
    Raises ValueError on unrecognised characters or wrong length.
    Leftmost character is the most significant trit.
    """
    s = s.strip().lower()
    if len(s) != 5:
        raise ValueError(f'pattern must be 5 characters, got {s!r}')
    code = 0
    for ch in s:
        if ch == 'g':
            code = code * 3 + 2
        elif ch == 'y':
            code = code * 3 + 1
        elif _is_gray_char(ch):
            code = code * 3 + 0
        else:
            raise ValueError(f'bad pattern char {ch!r} in {s!r}')
    return code


def fmt_pattern(code: int) -> str:
    """Format a response code as a 5-char string: g=green, y=yellow, -=gray."""
    chars = {0: '-', 1: 'y', 2: 'g'}
    digits = []
    for _ in range(5):
        digits.append(code % 3)
        code //= 3
    return ''.join(chars[d] for d in reversed(digits))
