"""`.env.example` must agree with the defaults it claims to document.

It did not. The file opened with "Every value below is the default in src/config.py" and
then set CHUNK_SIZE=600, CHUNK_OVERLAP=100 and TOP_K=4 against defaults of 800, 120 and 5,
and named a collection, `insurance_docs`, that nothing else in the repository uses.

That is not a cosmetic drift. Copying the file as its own first line instructs rebuilt the
index at 426 chunks instead of the 314 every published number was measured on, and 600/100
is specifically the superseded dense baseline the README shows the project moving away
from. The reproduce instructions would then produce numbers that disagree with the
documented ones, with nothing to say why.

These tests are the reason that cannot happen again silently.
"""

from pathlib import Path

import pytest

from src.config import Config

ROOT = Path(__file__).resolve().parents[1]
ENV_EXAMPLE = ROOT / ".env.example"

#: Settings frozen by the evaluation. eval/retrieval_config.json records the configuration
#: every published metric was measured at; these three define the index itself.
FROZEN = {"CHUNK_SIZE": "800", "CHUNK_OVERLAP": "120", "TOP_K": "5"}


def parse_env_example() -> dict[str, str]:
    """The KEY=VALUE pairs, keeping the value exactly as written."""
    settings: dict[str, str] = {}
    for raw in ENV_EXAMPLE.read_text(encoding="utf-8").splitlines():
        line = raw.lstrip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        settings[key.strip()] = _unquote(value)
    return settings


def _unquote(value: str) -> str:
    """What python-dotenv would hand the application for this right-hand side.

    Quoting is not decoration here: dotenv strips trailing whitespace from an unquoted
    value, and BGE_QUERY_PREFIX ends in a space.
    """
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        return value[1:-1]
    return value.strip()


def config_defaults() -> dict[str, str]:
    """Every setting Config exposes, as the string a .env file would carry."""
    return {
        name: str(getattr(Config, name))
        for name in dir(Config)
        if name.isupper() and not name.startswith("_")
    }


def test_the_file_is_parseable_and_not_empty():
    settings = parse_env_example()
    assert len(settings) >= 10, f"only {len(settings)} settings parsed from .env.example"


@pytest.mark.parametrize("key", sorted(FROZEN))
def test_the_frozen_retrieval_settings_are_the_evaluated_ones(key):
    """Guarded by name as well as by equality.

    If someone changes both config.py and .env.example together the equality test below
    still passes, and the index would silently stop matching eval/reference_run.json. These
    three are pinned to their literal values for that reason.
    """
    assert parse_env_example()[key] == FROZEN[key]
    assert str(getattr(Config, key)) == FROZEN[key]


def test_every_documented_value_equals_the_default():
    example, defaults = parse_env_example(), config_defaults()
    mismatched = {
        key: (value, defaults[key])
        for key, value in example.items()
        if key in defaults and value != defaults[key]
    }
    assert not mismatched, (
        "`.env.example` claims every value is the default in src/config.py, but these "
        "differ (file, default): " + repr(mismatched)
    )


def test_no_setting_is_documented_that_config_does_not_read():
    example, defaults = parse_env_example(), config_defaults()
    unknown = sorted(set(example) - set(defaults))
    assert not unknown, f".env.example documents settings src/config.py never reads: {unknown}"


def test_every_setting_config_reads_is_documented():
    example, defaults = parse_env_example(), config_defaults()
    undocumented = sorted(set(defaults) - set(example))
    assert not undocumented, (
        f"src/config.py reads settings .env.example does not document: {undocumented}"
    )


def test_the_example_carries_no_real_credential():
    """It is a template. Anything that looks like a populated secret is a mistake."""
    for key, value in parse_env_example().items():
        if any(marker in key for marker in ("KEY", "TOKEN", "SECRET", "PASSWORD")):
            assert value == "", f"{key} must ship empty, not {value!r}"


def test_dotenv_itself_reproduces_every_default():
    """The parser above approximates dotenv. This uses the real one.

    Copying `.env.example` to `.env` must leave the application on exactly the defaults it
    already had, or the file changes behaviour while claiming to document it. This is the
    test that catches quoting bugs: an unquoted BGE_QUERY_PREFIX loses its trailing space
    here and nowhere else.
    """
    from dotenv import dotenv_values

    loaded = dotenv_values(ENV_EXAMPLE)
    defaults = config_defaults()
    differing = {
        key: (value, defaults[key])
        for key, value in loaded.items()
        if key in defaults and value != defaults[key]
    }
    assert not differing, (
        "loading .env.example through python-dotenv does not reproduce the defaults "
        "(loaded, default): " + repr(differing)
    )


def test_the_query_prefix_keeps_its_trailing_space():
    """Named explicitly because it is invisible in a diff and changes every embedding."""
    from dotenv import dotenv_values

    prefix = dotenv_values(ENV_EXAMPLE)["BGE_QUERY_PREFIX"]
    assert prefix.endswith(" "), (
        "BGE_QUERY_PREFIX lost its trailing space, so queries would embed as "
        '"...passages:Who is covered?" instead of "...passages: Who is covered?"'
    )
    assert prefix == Config.BGE_QUERY_PREFIX
