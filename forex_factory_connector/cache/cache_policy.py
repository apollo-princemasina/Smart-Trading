from datetime import timedelta

# Hard TTLs — how long a cache entry is considered fresh
CALENDAR_THISWEEK_TTL = timedelta(minutes=5)
CALENDAR_NEXTWEEK_TTL = timedelta(minutes=10)
CALENDAR_LASTWEEK_TTL = timedelta(hours=1)

# Stale-serve windows — how long we continue serving an entry after it goes stale
CALENDAR_THISWEEK_STALE = timedelta(minutes=30)
CALENDAR_NEXTWEEK_STALE = timedelta(hours=4)
CALENDAR_LASTWEEK_STALE = timedelta(hours=24)
