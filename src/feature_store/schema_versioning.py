"""Semantic versioning for feature schemas.

Version format: MAJOR.MINOR.PATCH

* MAJOR — breaking structural changes (column removed / renamed / dtype changed)
* MINOR — backward-compatible additions (new columns)
* PATCH — metadata-only changes (description, author, tags)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from .schema import DatasetSchema, FeatureSchema


# ── SemanticVersion ───────────────────────────────────────────────────────────


@dataclass(frozen=True, order=True)
class SemanticVersion:
    """Immutable, comparable semantic version."""

    major: int
    minor: int
    patch: int

    # ── Parsing ───────────────────────────────────────────────────────────────

    @classmethod
    def parse(cls, version_str: str) -> SemanticVersion:
        """Parse ``"1.2.3"`` or ``"v1.2.3"``."""
        s = version_str.lstrip("v").strip()
        m = re.fullmatch(r"(\d+)\.(\d+)\.(\d+)", s)
        if not m:
            raise ValueError(
                f"Invalid semantic version {version_str!r}. "
                "Expected MAJOR.MINOR.PATCH (e.g. '1.0.0')."
            )
        return cls(int(m.group(1)), int(m.group(2)), int(m.group(3)))

    # ── Display ───────────────────────────────────────────────────────────────

    def __str__(self) -> str:
        return f"{self.major}.{self.minor}.{self.patch}"

    def __repr__(self) -> str:
        return f"SemanticVersion('{self}')"

    # ── Compatibility ─────────────────────────────────────────────────────────

    def is_compatible_with(self, other: SemanticVersion) -> bool:
        """Two schemas are compatible when they share the same MAJOR version."""
        return self.major == other.major

    def bumped_major(self) -> SemanticVersion:
        return SemanticVersion(self.major + 1, 0, 0)

    def bumped_minor(self) -> SemanticVersion:
        return SemanticVersion(self.major, self.minor + 1, 0)

    def bumped_patch(self) -> SemanticVersion:
        return SemanticVersion(self.major, self.minor, self.patch + 1)


# ── Version-bump logic ────────────────────────────────────────────────────────

BumpType = Literal["major", "minor", "patch", "none"]


def determine_version_bump(
    old_features: dict[str, FeatureSchema],
    new_features: dict[str, FeatureSchema],
) -> BumpType:
    """
    Compare two feature maps and return the required version bump type.

    Rules (evaluated in order of severity):
    1. Any existing feature removed           → MAJOR
    2. Any existing feature dtype changed     → MAJOR
    3. Any existing feature prefix changed    → MAJOR
    4. Any new feature added                  → MINOR
    5. Only metadata changed                  → PATCH
    6. No changes                             → none
    """
    old_names = set(old_features)
    new_names = set(new_features)

    removed = old_names - new_names
    if removed:
        return "major"

    # Check structural changes on existing features
    for name in old_names & new_names:
        old_f = old_features[name]
        new_f = new_features[name]
        if old_f.dtype != new_f.dtype or old_f.prefix != new_f.prefix:
            return "major"

    added = new_names - old_names
    if added:
        return "minor"

    # Check for any metadata-level changes
    for name in old_names & new_names:
        if old_features[name].to_dict() != new_features[name].to_dict():
            return "patch"

    return "none"


def next_version(
    current: str,
    old_features: dict[str, FeatureSchema],
    new_features: dict[str, FeatureSchema],
) -> str:
    """
    Compute the next schema version string given a before/after feature map.

    Returns the *current* version string unchanged if no bump is needed.
    """
    sv  = SemanticVersion.parse(current)
    bump = determine_version_bump(old_features, new_features)
    if bump == "major":
        return str(sv.bumped_major())
    if bump == "minor":
        return str(sv.bumped_minor())
    if bump == "patch":
        return str(sv.bumped_patch())
    return current


def sort_versions(versions: list[str]) -> list[str]:
    """Sort version strings in ascending order."""
    parsed = [(SemanticVersion.parse(v), v) for v in versions]
    return [v for _, v in sorted(parsed)]


def latest_version(versions: list[str]) -> str:
    """Return the highest version string from a list."""
    return sort_versions(versions)[-1]
