"""DFE bound logger — wraps loguru with module context."""
from loguru import logger as _base

logger = _base.bind(module="decision_fusion")
