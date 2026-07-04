"""Backward-compatibility checking between schema versions."""

from __future__ import annotations

from dataclasses import dataclass, field

from .exceptions import CompatibilityError
from .schema import DatasetSchema
from .schema_versioning import SemanticVersion, determine_version_bump


@dataclass
class CompatibilityReport:
    """Full result of a compatibility check between two schema versions."""

    is_compatible:          bool
    version_a:              str
    version_b:              str
    breaking_changes:       list[str] = field(default_factory=list)
    non_breaking_changes:   list[str] = field(default_factory=list)
    new_features:           list[str] = field(default_factory=list)
    removed_features:       list[str] = field(default_factory=list)
    type_changes:           dict[str, tuple[str, str]] = field(default_factory=dict)
    prefix_changes:         dict[str, tuple[str, str]] = field(default_factory=dict)

    def raise_if_incompatible(self) -> None:
        if not self.is_compatible:
            changes = "; ".join(self.breaking_changes[:5])
            raise CompatibilityError(
                f"Schema v{self.version_a} → v{self.version_b} has "
                f"{len(self.breaking_changes)} breaking change(s): {changes}"
            )

    def summary(self) -> str:
        status = "✓ Compatible" if self.is_compatible else "✗ INCOMPATIBLE"
        return (
            f"{status}: {self.version_a} → {self.version_b}  |  "
            f"{len(self.breaking_changes)} breaking, "
            f"{len(self.non_breaking_changes)} non-breaking, "
            f"{len(self.new_features)} new, "
            f"{len(self.removed_features)} removed"
        )


class CompatibilityChecker:
    """
    Check whether two :class:`DatasetSchema` versions are compatible.

    Compatibility rules
    -------------------
    A schema version *B* is **backward-compatible** with *A* when:

    * No feature present in *A* is removed in *B*.
    * No feature present in *A* changes its dtype or prefix in *B*.
    * Both versions share the same MAJOR semantic version number.

    New features added in *B* are non-breaking.
    """

    def check(
        self,
        schema_a: DatasetSchema,
        schema_b: DatasetSchema,
    ) -> CompatibilityReport:
        """
        Compare *schema_a* (older / training) with *schema_b* (newer / inference).

        Returns a :class:`CompatibilityReport`.
        """
        feats_a = schema_a.features
        feats_b = schema_b.features

        names_a = set(feats_a)
        names_b = set(feats_b)

        removed = sorted(names_a - names_b)
        added   = sorted(names_b - names_a)

        type_changes:   dict[str, tuple[str, str]] = {}
        prefix_changes: dict[str, tuple[str, str]] = {}

        for name in names_a & names_b:
            fa, fb = feats_a[name], feats_b[name]
            if fa.dtype != fb.dtype:
                type_changes[name]   = (fa.dtype,   fb.dtype)
            if fa.prefix != fb.prefix:
                prefix_changes[name] = (fa.prefix, fb.prefix)

        # Breaking: removed features, type changes, prefix changes, major bump
        breaking: list[str] = []
        for feat in removed:
            breaking.append(f"Feature removed: '{feat}'")
        for feat, (old_t, new_t) in type_changes.items():
            breaking.append(f"Dtype changed for '{feat}': {old_t} → {new_t}")
        for feat, (old_p, new_p) in prefix_changes.items():
            breaking.append(f"Prefix changed for '{feat}': {old_p} → {new_p}")

        # Major version change is itself a breaking signal
        ver_a = SemanticVersion.parse(schema_a.version)
        ver_b = SemanticVersion.parse(schema_b.version)
        if not ver_a.is_compatible_with(ver_b) and not breaking:
            breaking.append(
                f"Major version bump ({schema_a.version} → {schema_b.version}) "
                "indicates intentional breaking change"
            )

        non_breaking: list[str] = [f"Feature added: '{f}'" for f in added]

        return CompatibilityReport(
            is_compatible        = len(breaking) == 0,
            version_a            = schema_a.version,
            version_b            = schema_b.version,
            breaking_changes     = breaking,
            non_breaking_changes = non_breaking,
            new_features         = added,
            removed_features     = removed,
            type_changes         = type_changes,
            prefix_changes       = prefix_changes,
        )

    def is_backward_compatible(
        self,
        schema_a: DatasetSchema,
        schema_b: DatasetSchema,
    ) -> bool:
        """
        Return True if *schema_b* is backward-compatible with *schema_a*.

        A model trained on *schema_a* can run inference using data conforming
        to *schema_b* only when True.
        """
        return self.check(schema_a, schema_b).is_compatible

    def get_migration_plan(
        self,
        schema_a: DatasetSchema,
        schema_b: DatasetSchema,
    ) -> dict:
        """
        Return a structured migration plan to move from *schema_a* to *schema_b*.

        The plan lists actions required to align a dataset built with *schema_a*
        to the *schema_b* structure.
        """
        report = self.check(schema_a, schema_b)
        actions = []

        for feat in report.removed_features:
            actions.append({"action": "remove_column", "feature": feat})

        for feat in report.new_features:
            fb = schema_b.features[feat]
            actions.append({
                "action":       "add_column",
                "feature":      feat,
                "dtype":        fb.dtype,
                "nullable":     fb.nullable,
                "default":      fb.default_value,
            })

        for feat, (old_t, new_t) in report.type_changes.items():
            actions.append({
                "action":  "cast_column",
                "feature": feat,
                "from":    old_t,
                "to":      new_t,
            })

        return {
            "from_version":     schema_a.version,
            "to_version":       schema_b.version,
            "is_compatible":    report.is_compatible,
            "n_actions":        len(actions),
            "actions":          actions,
            "breaking_changes": report.breaking_changes,
        }
