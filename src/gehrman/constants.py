"""Fixed enumerations and defaults used across Gehrman.

Keeping the skill list, experience levels, and platforms here (not in
free-form config) so later tool arguments can reuse the same closed set.
Categorical values must stay enumerable: open strings are harder to
validate and easier to abuse as an injection surface.
"""

from __future__ import annotations

from enum import StrEnum

CONFIG_DIR_ENV = "GEHRMAN_CONFIG_DIR"
DATA_DIR_ENV = "GEHRMAN_DATA_DIR"

# Local Ollama defaults. The agent talks to Ollama's OpenAI-compatible
# endpoint and runs the model on the local machine.
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434/v1"
DEFAULT_OLLAMA_MODEL = "llama3.1"
DEFAULT_MAX_TOOL_STEPS = 6
DEFAULT_MAX_TOKENS_PER_MESSAGE = 8000
DEFAULT_TEXT_TRUNCATE = 1000
DEFAULT_TIME_UNIT = "hours"
DEFAULT_WEIGHTS_VERSION = "1"

# Public snapshots from arkadiyt/bounty-targets-data.
BOUNTY_TARGETS_DATA_BASE = (
    "https://raw.githubusercontent.com/arkadiyt/bounty-targets-data/main/data"
)


class Platform(StrEnum):
    HACKERONE = "hackerone"
    BUGCROWD = "bugcrowd"
    INTIGRITI = "intigriti"


CATALOG_URLS: dict[Platform, str] = {
    Platform.HACKERONE: f"{BOUNTY_TARGETS_DATA_BASE}/hackerone_data.json",
    Platform.BUGCROWD: f"{BOUNTY_TARGETS_DATA_BASE}/bugcrowd_data.json",
    Platform.INTIGRITI: f"{BOUNTY_TARGETS_DATA_BASE}/intigriti_data.json",
}


class Skill(StrEnum):
    WEB = "web"
    API = "api"


class AssetType(StrEnum):
    """Normalized scope type. Unknown catalog types become `other`."""

    WEB = "web"
    WILDCARD = "wildcard"
    API = "api"
    MOBILE = "mobile"
    OTHER = "other"


SKILL_VALUES = tuple(skill.value for skill in Skill)
PLATFORM_VALUES = tuple(platform.value for platform in Platform)
ASSET_TYPE_VALUES = tuple(asset_type.value for asset_type in AssetType)
