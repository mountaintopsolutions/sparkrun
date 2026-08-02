"""Tests for sparkrun.utils.text placeholder substitution."""

from __future__ import annotations

import pytest

from sparkrun.utils.text import substitute_placeholders


class TestSubstitutePlaceholders:
    """Tests for substitute_placeholders()."""

    def test_standalone_placeholders(self):
        """Plain ``{key}`` tokens are substituted from the mapping."""
        rendered = substitute_placeholders("--host {host} --port {port}", {"host": "0.0.0.0", "port": 8000})

        assert rendered == "--host 0.0.0.0 --port 8000"

    def test_unknown_key_left_verbatim(self):
        """An unresolved placeholder is restored as-is (documented behavior)."""
        assert substitute_placeholders("--x {nope}", {"port": 8000}) == "--x {nope}"

    def test_none_value_left_verbatim(self):
        """A key present but resolving to None is treated as unresolved."""
        assert substitute_placeholders("--x {k}", {"k": None}) == "--x {k}"

    def test_falsy_values_are_substituted(self):
        """0 / False / '' substitute — only None means 'unresolved'."""
        assert substitute_placeholders("{a}|{b}|{c}", {"a": 0, "b": False, "c": ""}) == "0|False|"

    def test_placeholder_nested_in_brace_escape(self):
        """A placeholder inside ``{{...}}`` JSON renders; the escapes survive.

        The regression this whole helper exists for: vpd's ``arg_substitute``
        matched from the opening ``{{`` through the placeholder's closing brace,
        treated the span as one unknown variable and restored it verbatim —
        swallowing the placeholder.
        """
        rendered = substitute_placeholders(
            '--speculative-config \'{{"method":"mtp","num_speculative_tokens":{num_speculative_tokens}}}\'',
            {"num_speculative_tokens": 1},
        )

        assert rendered == '--speculative-config \'{{"method":"mtp","num_speculative_tokens":1}}\''

    def test_placeholder_nested_in_bare_json_braces(self):
        """Single-brace JSON with an inner placeholder also renders."""
        rendered = substitute_placeholders('{"method":"mtp","n":{n}}', {"n": 2})

        assert rendered == '{"method":"mtp","n":2}'

    def test_brace_escapes_are_not_collapsed(self):
        """Collapsing ``{{``/``}}`` is the caller's job, not the substituter's."""
        assert substitute_placeholders("{{literal}}", {"literal": "X"}) == "{{literal}}"

    def test_escape_does_not_open_a_placeholder(self):
        """``{{`` is consumed as an escape, so ``{{key}`` is not a placeholder.

        Under the old regex this matched with ``key`` stripped out of the span
        and substituted, dropping one of the escape braces.
        """
        assert substitute_placeholders("{{key}", {"key": "X"}) == "{{key}"

    def test_json_without_placeholders_untouched(self):
        """A JSON blob whose contents match no key passes through unchanged."""
        text = "--config '{\"canvas_length\": 256}'"

        assert substitute_placeholders(text, {"port": 8000}) == text

    def test_adjacent_placeholders(self):
        """Back-to-back placeholders each substitute independently."""
        assert substitute_placeholders("{a}{b}", {"a": "1", "b": "2"}) == "12"

    def test_empty_braces_left_verbatim(self):
        """``{}`` has no key to look up and is passed through."""
        assert substitute_placeholders("echo {}", {}) == "echo {}"

    def test_multiline_template(self):
        """Placeholders render across line continuations."""
        rendered = substitute_placeholders("vllm serve {model} \\\n    --port {port}", {"model": "m", "port": 8000})

        assert rendered == "vllm serve m \\\n    --port 8000"

    @pytest.mark.parametrize("key", ["max-num-seqs", "some.key", "_leading"])
    def test_non_identifier_keys(self, key):
        """Keys are not restricted to Python identifiers."""
        assert substitute_placeholders("--x {%s}" % key, {key: "v"}) == "--x v"

    def test_accepts_saf_variables_chain(self):
        """Works with any object exposing a one-argument ``.get``."""
        from sparkrun.core.recipe import Recipe

        recipe = Recipe.from_dict({"name": "t", "model": "m", "runtime": "vllm", "defaults": {"port": 8000}})

        assert substitute_placeholders("{model}:{port}", recipe.build_config_chain()) == "m:8000"
