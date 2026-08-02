"""String ⇄ value parsing and formatting helpers."""

from __future__ import annotations

import re
from typing import Any

# Placeholder scanner for command templates.  The alternation order matters:
# a ``{{``/``}}`` brace escape is consumed *before* the placeholder branch can
# see the inner brace, so an escape never becomes the opening brace of a
# placeholder (and vice versa).
_PLACEHOLDER_RE = re.compile(r"\{\{|\}\}|\{([^{}]*)\}")


def substitute_placeholders(text: str, values: Any) -> str:
    """Substitute ``{key}`` placeholders from ``values`` in a single pass.

    Replaces vpd's ``arg_substitute``, whose ``\\{(.*?)\\}`` regex is blind to
    ``{{``/``}}`` brace escapes: given a placeholder nested inside escaped JSON
    braces it matched from the *escape* through the placeholder's closing brace,
    treated that whole span as one variable name, failed the lookup and restored
    it verbatim — silently swallowing the real placeholder.  A recipe line like::

        --speculative-config '{{"method":"mtp","num_speculative_tokens":{num_speculative_tokens}}}'

    therefore reached the runtime with ``{num_speculative_tokens}`` unrendered.
    Standalone placeholders (``{host}``, ``{port}``) were unaffected, which is
    why the failure only showed up on JSON-valued flags.

    Scanning left to right, each match is one of:

    - ``{{`` or ``}}`` — a brace escape, emitted verbatim.  Collapsing to a
      single brace is a separate, caller-controlled step (see
      ``recipe._collapse_brace_escapes``), so escaping survives the round trip.
    - ``{key}`` — substituted with ``str(values.get(key))``.  An unknown key (or
      one resolving to ``None``) is restored verbatim, matching the documented
      "unresolved placeholders are left as-is" behavior.

    Args:
        text: Template string.
        values: Anything with a one-argument ``.get(key)`` — a ``dict`` or a SAF
            ``Variables`` config chain.

    Returns:
        The rendered string.
    """

    def _replace(match: re.Match) -> str:
        key = match.group(1)
        if key is None:  # a {{ or }} brace escape, not a placeholder
            return match.group(0)
        value = values.get(key)
        return match.group(0) if value is None else str(value)

    return _PLACEHOLDER_RE.sub(_replace, text)


def coerce_value(value: str):
    """Coerce a string value to int, float, or bool where possible."""
    if value.lower() in ("true", "yes"):
        return True
    if value.lower() in ("false", "no"):
        return False
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


def parse_kv_output(output: str) -> dict[str, str]:
    """Parse key=value lines from script output.

    Lines starting with ``#`` are ignored. Leading/trailing whitespace
    on keys and values is stripped.

    Args:
        output: Raw stdout containing key=value lines.

    Returns:
        Dictionary of parsed key=value pairs.
    """
    result: dict[str, str] = {}
    for line in output.strip().splitlines():
        line = line.strip()
        if "=" in line and not line.startswith("#"):
            key, _, value = line.partition("=")
            result[key.strip()] = value.strip()
    return result


def parse_scoped_name(name: str) -> tuple[str | None, str]:
    """Parse ``@registry/lookup_name`` into ``(registry, lookup_name)``.

    Returns ``(None, name)`` when the input has no ``@`` prefix or
    no ``/`` separator.
    """
    if name.startswith("@") and "/" in name:
        prefix, lookup_name = name.split("/", 1)
        return prefix[1:], lookup_name  # strip leading @
    return None, name


def format_duration(seconds: float) -> str:
    """Format a duration in seconds to a human-readable string.

    Returns ``"Xs"`` for durations under 60s, ``"Xm Ys"`` for durations
    under an hour, and ``"Xh Ym Zs"`` for longer durations.
    """
    s = int(seconds)
    if s < 60:
        return "%.1fs" % seconds
    m, s = divmod(s, 60)
    if m < 60:
        return "%dm %ds" % (m, s)
    h, m = divmod(m, 60)
    return "%dh %dm %ds" % (h, m, s)
