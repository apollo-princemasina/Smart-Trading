"""EventType enum — granular classification used by the EIE pipeline."""
from enum import Enum


class EventType(str, Enum):
    # Employment
    EMPLOYMENT = "EMPLOYMENT"           # NFP, payrolls, jobs reports
    UNEMPLOYMENT = "UNEMPLOYMENT"       # Unemployment Rate
    JOBLESS_CLAIMS = "JOBLESS_CLAIMS"   # Initial / Continuing Jobless Claims
    WAGES = "WAGES"                     # Average Hourly Earnings, wage growth

    # Inflation
    INFLATION = "INFLATION"             # CPI, PCE, PPI, RPI

    # Central Bank
    INTEREST_RATE = "INTEREST_RATE"     # Rate decisions (FOMC, ECB, BOE)
    CENTRAL_BANK_SPEECH = "CENTRAL_BANK_SPEECH"  # Powell, Lagarde, Bailey speeches

    # Growth
    GDP = "GDP"                         # GDP reports
    RETAIL_SALES = "RETAIL_SALES"       # Retail Sales
    CONSUMER_CONFIDENCE = "CONSUMER_CONFIDENCE"  # Consumer Sentiment / Confidence

    # Manufacturing & Activity
    PMI = "PMI"                         # Purchasing Managers Index
    MANUFACTURING = "MANUFACTURING"     # ISM Manufacturing, Factory Orders
    INDUSTRIAL = "INDUSTRIAL"           # Industrial Production

    # Trade
    TRADE_BALANCE = "TRADE_BALANCE"     # Trade Balance, Current Account

    # Housing
    HOUSING = "HOUSING"                 # Housing Starts, Existing Home Sales

    # Commodities
    OIL_INVENTORY = "OIL_INVENTORY"    # EIA Crude Oil Inventories

    # Misc
    POLITICAL = "POLITICAL"            # Elections, referenda
    HOLIDAY = "HOLIDAY"                # Market holidays
    UNKNOWN = "UNKNOWN"                # Could not classify
