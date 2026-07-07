from enum import Enum


class ImpactLevel(str, Enum):
    HIGH         = "HIGH"
    MEDIUM       = "MEDIUM"
    LOW          = "LOW"
    HOLIDAY      = "HOLIDAY"
    NON_ECONOMIC = "NON_ECONOMIC"


class EventStatus(str, Enum):
    SCHEDULED = "SCHEDULED"   # future event, no actual yet
    RELEASED  = "RELEASED"    # actual value is present
    REVISED   = "REVISED"     # previous value was revised after release
    CANCELLED = "CANCELLED"   # event did not occur


class EventCategory(str, Enum):
    EMPLOYMENT   = "EMPLOYMENT"
    INFLATION    = "INFLATION"
    GDP          = "GDP"
    TRADE        = "TRADE"
    CENTRAL_BANK = "CENTRAL_BANK"
    HOUSING      = "HOUSING"
    MANUFACTURING= "MANUFACTURING"
    RETAIL       = "RETAIL"
    SENTIMENT    = "SENTIMENT"
    SPEECH       = "SPEECH"
    OTHER        = "OTHER"


class Provider(str, Enum):
    FOREX_FACTORY     = "forex_factory"
    FXSTREET          = "fxstreet"          # reserved
    TRADING_ECONOMICS = "trading_economics" # reserved
    INVESTING_COM     = "investing_com"     # reserved
