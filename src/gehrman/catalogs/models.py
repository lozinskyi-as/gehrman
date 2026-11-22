"""Pydantic models for catalog payloads.

The three platforms ship different JSON shapes. Per-platform `Raw*Program`
models validate only the fields Gehrman actually reads — `extra="ignore"`
drops the rest. The unified `Program` is internal and uses `extra="forbid"`
so a typo in a field name is caught during development.

Security: free-text fields (`policy`, descriptions) are kept because the
rest of the pipeline needs them, but they stay untrusted until sanitized
and data-marked downstream. Never render them with Rich markup.
"""

from __future__ import annotations

from pydantic import AliasChoices, BaseModel, ConfigDict, Field

from gehrman.constants import AssetType, Platform


class RawTarget(BaseModel):
    """One scope entry. Fields are a superset across platforms."""

    model_config = ConfigDict(extra="ignore")

    asset_type: str | None = None
    asset_identifier: str | None = None
    eligible_for_bounty: bool | None = None
    type: str | None = None
    # Bugcrowd identifier, in priority order target -> uri -> name.
    target: str | None = None
    uri: str | None = None
    name: str | None = None
    # Intigriti identifier.
    endpoint: str | None = None


class RawTargets(BaseModel):
    model_config = ConfigDict(extra="ignore")

    in_scope: list[RawTarget] = Field(default_factory=list)
    out_of_scope: list[RawTarget] = Field(default_factory=list)


class HackerOneProgram(BaseModel):
    model_config = ConfigDict(extra="ignore")

    handle: str
    name: str
    url: str
    submission_state: str | None = None
    offers_bounties: bool = False
    policy: str = ""
    # The catalog key is average_time_to_first_program_response. The stored
    # column keeps the shorter name. Both keys validate so fixtures written
    # against either spelling still load.
    average_time_to_first_response: float | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "average_time_to_first_program_response",
            "average_time_to_first_response",
        ),
    )
    average_time_to_bounty_awarded: float | None = None
    average_time_to_report_resolved: float | None = None
    response_efficiency_percentage: float | None = None
    targets: RawTargets = Field(default_factory=RawTargets)


class BugcrowdProgram(BaseModel):
    model_config = ConfigDict(extra="ignore")

    name: str
    url: str
    max_payout: float | None = None
    allows_disclosure: bool | None = None
    safe_harbor: str | None = None
    targets: RawTargets = Field(default_factory=RawTargets)


class IntigritiBountyRange(BaseModel):
    model_config = ConfigDict(extra="ignore")

    value: float | None = None
    currency: str | None = None


class IntigritiProgram(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str | None = None
    name: str
    url: str
    min_bounty: IntigritiBountyRange | None = None
    max_bounty: IntigritiBountyRange | None = None
    # The catalog uses camelCase. Both spellings validate.
    two_factor_required: bool | None = Field(
        default=None,
        validation_alias=AliasChoices("twoFactorRequired", "two_factor_required"),
    )
    tac_required: bool | None = Field(
        default=None,
        validation_alias=AliasChoices("tacRequired", "tac_required"),
    )
    targets: RawTargets = Field(default_factory=RawTargets)


class ScopeAsset(BaseModel):
    """One structured scope entry. Free-text target fields are not stored."""

    model_config = ConfigDict(extra="forbid")

    identifier: str
    type: AssetType
    bounty_eligible: bool | None = None


class Program(BaseModel):
    """Unified, storage-ready program record."""

    model_config = ConfigDict(extra="forbid")

    platform: Platform
    handle: str
    name: str
    url: str
    offers_bounty: bool
    bounty_min: float | None = None
    bounty_max: float | None = None
    bounty_min_currency: str | None = None
    bounty_max_currency: str | None = None
    safe_harbor: str | None = None
    allows_disclosure: bool | None = None
    two_factor_required: bool | None = None
    tac_required: bool | None = None
    asset_types: list[str] = Field(default_factory=list)
    in_scope_assets: list[ScopeAsset] = Field(default_factory=list)
    out_of_scope_assets: list[ScopeAsset] = Field(default_factory=list)
    in_scope_count: int = 0
    out_of_scope_count: int = 0
    submission_state: str | None = None
    average_time_to_first_response: float | None = None
    average_time_to_bounty_awarded: float | None = None
    average_time_to_report_resolved: float | None = None
    response_efficiency_percentage: float | None = None
    policy_excerpt: str = ""
