"""Tests for sparkrun.utils.text placeholder substitution.

These lock down the templater's edge cases: brace escapes, JSON-valued
arguments, shell idioms that merely *look* like placeholders, and value
coercion.  The behavior under test is a single left-to-right scan whose
alternation consumes ``{{``/``}}`` before the placeholder branch can see the
inner brace.
"""

from __future__ import annotations

import logging

import pytest

from sparkrun.utils.text import render_template, substitute_placeholders


class TestBasicSubstitution:
    """Plain ``{key}`` replacement."""

    def test_standalone_placeholders(self):
        """Plain ``{key}`` tokens are substituted from the mapping."""
        rendered = substitute_placeholders("--host {host} --port {port}", {"host": "0.0.0.0", "port": 8000})

        assert rendered == "--host 0.0.0.0 --port 8000"

    def test_unknown_key_left_verbatim(self):
        """An unresolved placeholder is restored as-is (documented behavior)."""
        assert substitute_placeholders("--x {nope}", {"port": 8000}) == "--x {nope}"

    def test_adjacent_placeholders(self):
        """Back-to-back placeholders each substitute independently."""
        assert substitute_placeholders("{a}{b}", {"a": "1", "b": "2"}) == "12"

    def test_repeated_placeholder(self):
        """The same key may appear many times."""
        assert substitute_placeholders("{p}-{p}-{p}", {"p": "x"}) == "x-x-x"

    def test_no_placeholders_is_identity(self):
        """A template with nothing to substitute is returned unchanged."""
        assert substitute_placeholders("vllm serve model --trust-remote-code", {"a": 1}) == "vllm serve model --trust-remote-code"

    def test_empty_string(self):
        """The empty template is handled without error."""
        assert substitute_placeholders("", {"a": 1}) == ""

    def test_multiline_template(self):
        """Placeholders render across backslash line continuations."""
        rendered = substitute_placeholders("vllm serve {model} \\\n    --port {port}", {"model": "m", "port": 8000})

        assert rendered == "vllm serve m \\\n    --port 8000"

    def test_key_is_matched_case_sensitively(self):
        """``{PORT}`` and ``{port}`` are different keys."""
        assert substitute_placeholders("{PORT}/{port}", {"port": 8000}) == "{PORT}/8000"

    @pytest.mark.parametrize("key", ["max-num-seqs", "some.key", "_leading", "a1", "with space"])
    def test_non_identifier_keys(self, key):
        """Keys are not restricted to Python identifiers.

        The config chain is fed by ``-o key=value`` and recipe ``defaults``,
        neither of which constrains the key spelling, so the scanner must not
        either.
        """
        assert substitute_placeholders("--x {%s}" % key, {key: "v"}) == "--x v"


class TestValueTypes:
    """How a resolved value is rendered into the template."""

    def test_falsy_values_are_substituted(self):
        """0 / False / '' substitute — only None means 'unresolved'."""
        assert substitute_placeholders("{a}|{b}|{c}", {"a": 0, "b": False, "c": ""}) == "0|False|"

    def test_none_value_left_verbatim(self):
        """A key present but resolving to None is treated as unresolved."""
        assert substitute_placeholders("--x {k}", {"k": None}) == "--x {k}"

    def test_numeric_values_are_stringified(self):
        """int/float values render via str()."""
        assert substitute_placeholders("{i} {f}", {"i": 8000, "f": 0.85}) == "8000 0.85"

    def test_value_containing_spaces(self):
        """Values are inserted verbatim — no quoting is added."""
        assert substitute_placeholders("--x {k}", {"k": "a b"}) == "--x a b"

    def test_value_containing_braces_is_not_rescanned(self):
        """Substitution is single-pass: an inserted value is not re-scanned.

        Iterating to a fixpoint is the caller's job (``render_command``), which
        is what makes nested references like ``http://host:{port}`` work.
        """
        assert substitute_placeholders("{k}", {"k": "{other}", "other": "X"}) == "{other}"

    def test_accepts_saf_variables_chain(self):
        """Works with any object exposing a one-argument ``.get``."""
        from sparkrun.core.recipe import Recipe

        recipe = Recipe.from_dict({"name": "t", "model": "m", "runtime": "vllm", "defaults": {"port": 8000}})

        assert substitute_placeholders("{model}:{port}", recipe.build_config_chain()) == "m:8000"


class TestBraceEscapes:
    """``{{`` / ``}}`` are escapes and must never open or close a placeholder."""

    def test_brace_escapes_are_not_collapsed(self):
        """Collapsing ``{{``/``}}`` is the caller's job, not the substituter's."""
        assert substitute_placeholders("{{literal}}", {"literal": "X"}) == "{{literal}}"

    def test_escape_does_not_open_a_placeholder(self):
        """``{{`` is consumed as an escape, so ``{{key}`` is not a placeholder.

        Under the old regex this matched with ``key`` stripped out of the span
        and substituted, dropping one of the escape braces.
        """
        assert substitute_placeholders("{{key}", {"key": "X"}) == "{{key}"

    def test_placeholder_may_be_wrapped_in_escapes(self):
        """``{{{key}}}`` is an escaped brace, a placeholder, then an escape."""
        assert substitute_placeholders("{{{key}}}", {"key": "X"}) == "{{X}}"

    def test_trailing_single_brace_after_placeholder(self):
        """``{key}}`` substitutes and leaves the odd brace as a literal."""
        assert substitute_placeholders("{key}}", {"key": "X"}) == "X}"

    def test_odd_length_brace_runs(self):
        """An unpaired brace run is left alone rather than half-consumed."""
        assert substitute_placeholders("{{{", {"a": "X"}) == "{{{"
        assert substitute_placeholders("}}}", {"a": "X"}) == "}}}"

    def test_empty_escape_pair(self):
        """``{{}}`` is two escapes, not an empty placeholder."""
        assert substitute_placeholders("{{}}", {"": "X"}) == "{{}}"

    def test_empty_braces_left_verbatim(self):
        """``{}`` has no key to look up and is passed through."""
        assert substitute_placeholders("echo {}", {}) == "echo {}"

    def test_escapes_and_placeholders_interleaved(self):
        """Escapes between placeholders don't disturb either."""
        assert substitute_placeholders("{a}{{b}}{c}", {"a": "1", "b": "3", "c": "2"}) == "1{{b}}2"


class TestJsonValuedArgs:
    """The failure mode this templater exists to fix."""

    def test_placeholder_nested_in_brace_escape(self):
        """A placeholder inside ``{{...}}`` JSON renders; the escapes survive.

        The regression: vpd's ``arg_substitute`` matched from the opening ``{{``
        through the placeholder's closing brace, treated the span as one unknown
        variable and restored it verbatim — swallowing the placeholder, so vLLM
        received a literal ``{num_speculative_tokens}`` and rejected the value.
        """
        rendered = substitute_placeholders(
            '--speculative-config \'{{"method":"mtp","num_speculative_tokens":{num_speculative_tokens}}}\'',
            {"num_speculative_tokens": 1},
        )

        assert rendered == '--speculative-config \'{{"method":"mtp","num_speculative_tokens":1}}\''

    def test_placeholder_nested_in_bare_json_braces(self):
        """Single-brace JSON with an inner placeholder also renders."""
        assert substitute_placeholders('{"method":"mtp","n":{n}}', {"n": 2}) == '{"method":"mtp","n":2}'

    def test_placeholder_nested_in_nested_json(self):
        """Nesting depth doesn't matter — only the placeholder is touched."""
        assert substitute_placeholders('{"x":{"y":{n}}}', {"n": 1}) == '{"x":{"y":1}}'

    def test_json_without_placeholders_untouched(self):
        """A JSON blob whose contents match no key passes through unchanged."""
        text = "--config '{\"canvas_length\": 256}'"

        assert substitute_placeholders(text, {"port": 8000}) == text

    def test_json_null_value_untouched(self):
        """``{"max_new_tokens": null}`` is not mistaken for a placeholder."""
        text = '{"max_new_tokens": null}'

        assert substitute_placeholders(text, {"max_new_tokens": 4}) == text

    def test_json_array_value_untouched(self):
        """Brackets and quotes inside JSON are irrelevant to the scanner."""
        text = '{"cudagraph_mode":"FULL_AND_PIECEWISE","custom_ops":["all"]}'

        assert substitute_placeholders(text, {"port": 8000}) == text


class TestRenderTemplate:
    """The bounded fixpoint loop around ``substitute_placeholders``."""

    def test_resolves_nested_reference(self):
        """A value containing a placeholder is resolved by a later pass."""
        assert render_template("{base_url}", {"base_url": "http://h:{port}", "port": 8000}) == "http://h:8000"

    def test_resolves_deep_chain(self):
        """Chained references resolve however deep, within the bound."""
        assert render_template("{a}", {"a": "{b}", "b": "{c}", "c": "end"}) == "end"

    def test_self_resolving_value_is_a_fixpoint(self):
        """``a: "{a}"`` stabilizes on the first pass — not a cycle."""
        assert render_template("{a}", {"a": "{a}"}) == "{a}"

    def test_self_growing_value_terminates(self, caplog):
        """``a: "x{a}"`` grows every pass; the loop must stop and say so.

        Without the bound this never reaches a fixpoint and the render hangs
        while the string grows without limit.
        """
        with caplog.at_level(logging.WARNING, logger="sparkrun.utils.text"):
            rendered = render_template("{a}", {"a": "x{a}"})

        assert rendered == "x" * 10 + "{a}"
        assert "did not stabilize" in caplog.text

    def test_mutual_recursion_terminates(self, caplog):
        """Two values referencing each other also hit the bound rather than hang."""
        with caplog.at_level(logging.WARNING, logger="sparkrun.utils.text"):
            rendered = render_template("{a}", {"a": "1{b}", "b": "2{a}"})

        assert rendered.endswith("{a}") or rendered.endswith("{b}")
        assert "did not stabilize" in caplog.text

    def test_max_passes_is_configurable(self):
        """The bound is a keyword argument, not a hard-coded constant."""
        assert render_template("{a}", {"a": "x{a}"}, max_passes=3) == "xxx{a}"

    def test_no_warning_for_ordinary_templates(self, caplog):
        """A template that stabilizes logs nothing."""
        with caplog.at_level(logging.WARNING, logger="sparkrun.utils.text"):
            render_template("--port {port}", {"port": 8000})

        assert caplog.text == ""


class TestShellIdioms:
    """Text that looks like a placeholder but is shell syntax."""

    def test_awk_program_untouched(self):
        """``awk '{print $1}'`` has no key named ``print $1``."""
        text = "docker ps | awk '{print $1}'"

        assert substitute_placeholders(text, {"port": 8000}) == text

    def test_shell_parameter_expansion_without_matching_key(self):
        """``${VAR}`` is left alone when VAR is not in the config chain."""
        assert substitute_placeholders("echo ${HF_HOME}", {"port": 8000}) == "echo ${HF_HOME}"

    def test_shell_parameter_expansion_with_matching_key_is_substituted(self):
        """``${VAR}`` IS rewritten when VAR happens to be a config key.

        Known sharp edge, unchanged from the previous renderer: the scanner has
        no notion of a preceding ``$``.  Locked down so any future change to it
        is a deliberate one rather than an accident.
        """
        assert substitute_placeholders("echo ${HOME}", {"HOME": "/root"}) == "echo $/root"

    def test_shell_default_expansion_untouched(self):
        """``${VAR:-default}`` contains a colon, so it can't match a bare key."""
        text = "echo ${PORT:-8000}"

        assert substitute_placeholders(text, {"PORT": 9000}) == text

    def test_newline_inside_braces_is_not_a_placeholder(self):
        """A brace pair spanning a newline is not treated as one key."""
        text = "{a\nb}"

        assert substitute_placeholders(text, {"a": "X", "b": "Y"}) == text
