"""FeatureRegistry — central hub for discovering and ordering feature generators.

Registration
------------
Feature classes self-register via the ``@FeatureRegistry.register`` decorator.
When each category subpackage is imported (triggered by ``src/features/__init__.py``),
every ``@FeatureRegistry.register``-decorated class definition fires the decorator,
which inserts the class into the shared ``_features`` dict.

Execution ordering
------------------
``get_execution_order()`` runs a topological sort (Kahn's BFS algorithm) on the
dependency graph so that a feature with dependencies always runs after its
prerequisites.  Circular dependencies raise ``ValueError`` immediately.

Thread safety
-------------
The registry is populated at import time (module-level class definitions) before
any concurrent execution starts.  Write access (register / enable / disable) is
therefore safe without locks in the typical usage pattern.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import ClassVar, Iterator

from .base_feature import BaseFeature

logger = logging.getLogger(__name__)


class FeatureRegistry:
    """Singleton-style class registry for all feature generators.

    All state lives in class variables so every import of this module
    shares the same registry instance.
    """

    _features: ClassVar[dict[str, type[BaseFeature]]] = {}
    _disabled: ClassVar[set[str]]                     = set()

    # ── Registration ────────────────────────────────────────────────────────

    @classmethod
    def register(cls, feature_class: type[BaseFeature]) -> type[BaseFeature]:
        """Class decorator that inserts *feature_class* into the registry.

        Usage::

            @FeatureRegistry.register
            class MyFeature(BaseFeature):
                name = "my_feature"
                ...

        Parameters
        ----------
        feature_class:
            A concrete subclass of ``BaseFeature`` with a non-empty ``name``.

        Returns
        -------
        type[BaseFeature]
            The same class, unmodified (decorator passthrough).

        Raises
        ------
        TypeError
            If *feature_class* is not a subclass of ``BaseFeature``.
        ValueError
            If ``feature_class.name`` is empty or blank.
        """
        if not issubclass(feature_class, BaseFeature):
            raise TypeError(
                f"Only subclasses of BaseFeature can be registered. "
                f"Got: {feature_class!r}"
            )

        name = feature_class.name.strip()
        if not name:
            raise ValueError(
                f"{feature_class.__name__}.name must be a non-empty string."
            )

        if name in cls._features:
            existing = cls._features[name].__module__
            logger.warning(
                "Feature '%s' is already registered (from %s). "
                "Overwriting with %s.",
                name, existing, feature_class.__module__,
            )

        cls._features[name] = feature_class
        logger.debug(
            "Registered feature: %s  category=%s  deps=%s",
            name, feature_class.category, feature_class.dependencies,
        )
        return feature_class

    # ── Registry queries ────────────────────────────────────────────────────

    @classmethod
    def all_features(cls) -> dict[str, type[BaseFeature]]:
        """Return every registered feature, regardless of enabled state."""
        return dict(cls._features)

    @classmethod
    def enabled_features(cls) -> dict[str, type[BaseFeature]]:
        """Return only features that are not in the disabled set."""
        return {
            name: fc
            for name, fc in cls._features.items()
            if name not in cls._disabled
        }

    @classmethod
    def get(cls, name: str) -> type[BaseFeature]:
        """Retrieve a registered feature class by name.

        Raises
        ------
        KeyError
            If *name* is not registered.
        """
        if name not in cls._features:
            registered = sorted(cls._features.keys())
            raise KeyError(
                f"Feature '{name}' not found in registry. "
                f"Registered features: {registered}"
            )
        return cls._features[name]

    @classmethod
    def get_instance(cls, name: str) -> BaseFeature:
        """Return a fresh instance of the named feature class."""
        return cls.get(name)()

    @classmethod
    def features_by_category(cls) -> dict[str, list[str]]:
        """Return a mapping of category → list of feature names."""
        result: dict[str, list[str]] = defaultdict(list)
        for name, fc in cls._features.items():
            result[fc.category].append(name)
        return dict(result)

    @classmethod
    def iter_enabled(cls) -> Iterator[BaseFeature]:
        """Yield fresh instances of all enabled features in dependency order."""
        for name in cls.get_execution_order():
            yield cls.get_instance(name)

    # ── Enable / disable ────────────────────────────────────────────────────

    @classmethod
    def disable(cls, name: str) -> None:
        """Exclude *name* from future ``get_execution_order()`` calls."""
        if name not in cls._features:
            raise KeyError(f"Feature '{name}' is not registered.")
        cls._disabled.add(name)
        logger.info("Feature disabled: %s", name)

    @classmethod
    def enable(cls, name: str) -> None:
        """Re-include *name* in the execution order."""
        cls._disabled.discard(name)
        logger.info("Feature enabled: %s", name)

    @classmethod
    def disable_category(cls, category: str) -> None:
        """Disable every registered feature in *category*."""
        for name, fc in cls._features.items():
            if fc.category == category:
                cls._disabled.add(name)
        logger.info("All features in category '%s' disabled.", category)

    # ── Execution ordering ──────────────────────────────────────────────────

    @classmethod
    def get_execution_order(cls) -> list[str]:
        """Return enabled feature names sorted by dependency (topological).

        Uses Kahn's BFS algorithm.  Raises ``ValueError`` on cycles.
        """
        enabled = cls.enabled_features()
        return cls._topological_sort(list(enabled.keys()))

    @classmethod
    def _topological_sort(cls, names: list[str]) -> list[str]:
        """Kahn's algorithm — O(V + E) topological sort.

        Only considers dependencies that are *also* in *names* (enabled
        features).  Dependencies on disabled or unregistered features are
        ignored (they produce a warning).
        """
        name_set = set(names)
        in_degree: dict[str, int]        = {n: 0 for n in names}
        adj:       dict[str, list[str]]  = {n: [] for n in names}

        for name in names:
            feature_cls = cls._features[name]
            for dep in feature_cls.dependencies:
                if dep not in name_set:
                    logger.warning(
                        "Feature '%s' depends on '%s', which is not in the "
                        "enabled set. Dependency ignored for this run.",
                        name, dep,
                    )
                    continue
                in_degree[name] += 1
                adj[dep].append(name)

        # BFS: start with every zero-in-degree node (sorted for determinism)
        queue:  list[str] = sorted(n for n in names if in_degree[n] == 0)
        result: list[str] = []

        while queue:
            node = queue.pop(0)
            result.append(node)
            for neighbor in sorted(adj[node]):
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if len(result) != len(names):
            cycle_nodes = [n for n in names if n not in result]
            raise ValueError(
                f"Circular dependency detected among features: {cycle_nodes}. "
                "Review the 'dependencies' attributes of those feature classes."
            )

        return result

    # ── Reporting ───────────────────────────────────────────────────────────

    @classmethod
    def summary(cls) -> str:
        """Return a human-readable summary of the registry state."""
        total    = len(cls._features)
        enabled  = total - len(cls._disabled)
        by_cat   = cls.features_by_category()

        lines = [
            f"FeatureRegistry — {total} registered, {enabled} enabled",
            "",
        ]
        for cat in sorted(by_cat):
            names = by_cat[cat]
            entries = []
            for n in sorted(names):
                status = "disabled" if n in cls._disabled else "enabled"
                entries.append(f"    {n} [{status}]")
            lines.append(f"  {cat} ({len(names)}):")
            lines.extend(entries)

        return "\n".join(lines)

    @classmethod
    def reset(cls) -> None:
        """Clear the registry.  Used in tests only — not for production use."""
        cls._features.clear()
        cls._disabled.clear()
        logger.warning("FeatureRegistry has been reset.")
