"""Tests for the round-trip INI editor."""

from __future__ import annotations

from llama_buddy.ini import IniFile

_HAND_WRITTEN = """\
version = 1

; shared by every model
[*]
c = 8192  ; plenty
LLAMA_ARG_CACHE_RAM = 0

# coding model
[laguna]
model = /models/laguna-s-2.1-Q4_K_M.gguf
temp = 0.7
"""


def test_untouched_file_round_trips():
    assert IniFile(_HAND_WRITTEN).text() == _HAND_WRITTEN


def test_reads_like_llama_cpp():
    ini = IniFile(_HAND_WRITTEN)
    assert ini.sections() == ["*", "laguna"]
    assert ini.get(None, "version") == "1"
    assert ini.get("*", "c") == "8192"  # inline comment stripped
    assert ini.get("*", "LLAMA_ARG_CACHE_RAM") == "0"  # case kept
    assert ini.get("*", "llama_arg_cache_ram") is None
    assert ini.get("laguna", "missing", "x") == "x"
    assert ini.get("nope", "temp") is None


def test_set_changes_only_its_line():
    ini = IniFile(_HAND_WRITTEN)
    ini.set("laguna", "temp", "0.6")
    assert ini.text() == _HAND_WRITTEN.replace("temp = 0.7", "temp = 0.6")


def test_set_appends_new_key_before_section_gap():
    ini = IniFile(_HAND_WRITTEN)
    ini.set("*", "fa", "on")
    assert "LLAMA_ARG_CACHE_RAM = 0\nfa = on\n\n# coding model" in ini.text()


def test_set_adds_missing_section_at_end():
    ini = IniFile(_HAND_WRITTEN)
    ini.set("org/m-GGUF:Q8_0", "spec-type", "draft-mtp")
    assert ini.text().endswith(
        "temp = 0.7\n\n[org/m-GGUF:Q8_0]\nspec-type = draft-mtp\n"
    )


def test_set_collapses_duplicates_to_the_last():
    ini = IniFile("[m]\nc = 1\nfa = on\nc = 2\n")
    assert ini.get("m", "c") == "2"  # llama.cpp keeps the last one
    ini.set("m", "c", "3")
    assert ini.text() == "[m]\nfa = on\nc = 3\n"


def test_remove_option_and_section():
    ini = IniFile(_HAND_WRITTEN)
    assert ini.remove_option("*", "c") is True
    assert ini.remove_option("*", "c") is False
    assert ini.remove_section("laguna") is True
    assert ini.sections() == ["*"]
    assert "laguna" not in ini.text()


def test_top_level_keys_stay_above_sections():
    ini = IniFile("[*]\nc = 0\n")
    ini.set(None, "port", "9090")
    assert ini.text() == "port = 9090\n[*]\nc = 0\n"


def test_empty_document():
    ini = IniFile()
    assert ini.text() == ""
    ini.set("m", "c", "1")
    assert ini.text() == "[m]\nc = 1\n"


def test_get_any_takes_first_spelling_present():
    ini = IniFile("[m]\nctx-size = 4096\n")
    assert ini.get_any("m", ("c", "ctx-size")) == "4096"
    assert ini.get_any("m", ("fa", "flash-attn"), "auto") == "auto"
