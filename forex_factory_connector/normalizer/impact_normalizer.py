from ..models.enums import ImpactLevel

_MAP: dict[str, ImpactLevel] = {
    "high":          ImpactLevel.HIGH,
    "medium":        ImpactLevel.MEDIUM,
    "low":           ImpactLevel.LOW,
    "holiday":       ImpactLevel.HOLIDAY,
    "non-economic":  ImpactLevel.NON_ECONOMIC,
    "noneconomic":   ImpactLevel.NON_ECONOMIC,
    "":              ImpactLevel.NON_ECONOMIC,
}


def normalize_impact(raw: str) -> ImpactLevel:
    return _MAP.get(raw.strip().lower(), ImpactLevel.NON_ECONOMIC)
