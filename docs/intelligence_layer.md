# MFIP Market Intelligence Layer
## Architecture & Operational Reference

---

## Overview

The Market Intelligence Layer is a production-grade background service that continuously ingests, normalises, caches, and exposes economic calendar intelligence from Forex Factory to the rest of the MFIP platform.

It is designed as a **pull-and-cache** architecture: all outbound requests to Forex Factory are initiated by a background scheduler, never by frontend or API requests. Every public endpoint reads exclusively from an in-memory cache.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        MFIP Backend                             │
│                                                                 │
│  ┌───────────────┐    ┌──────────────────────────────────────┐  │
│  │  FastAPI App  │    │     ForexFactoryConnector            │  │
│  │  (lifespan)   │───▶│  startup() → warm cache → scheduler │  │
│  └───────────────┘    └──────────────┬───────────────────────┘  │
│                                      │                          │
│         ┌────────────────────────────┼───────────────────────┐  │
│         ▼                            ▼                        ▼  │
│  ┌─────────────┐   ┌──────────────────────┐   ┌───────────────┐ │
│  │  REST API   │   │    APScheduler       │   │ MemoryCache   │ │
│  │ /intelligence│◀──│  CalendarJob (5min) │──▶│ (MFIPEvent[]) │ │
│  └─────────────┘   │  SpeechesJob (10min)│   └───────────────┘ │
│         │          │  NewsJob (2min)      │          ▲          │
│         │          │  SentimentJob (5min) │          │          │
│         │          └──────────┬───────────┘          │          │
│         │                     │                      │          │
│         ▼                     ▼                      │          │
│  ┌─────────────┐   ┌──────────────────────┐          │          │
│  │  Frontend   │   │    CDNFetcher        │          │          │
│  │  Next.js    │   │  GET nfs.faireconomy │          │          │
│  └─────────────┘   │  .media/ff_calendar_ │          │          │
│                    │  thisweek.json       │          │          │
│                    └──────────┬───────────┘          │          │
│                               │                      │          │
│                    ┌──────────▼───────────┐          │          │
│                    │  Parser → Validator  │──────────┘          │
│                    │  → Normalizer        │                     │
│                    └──────────────────────┘                     │
└─────────────────────────────────────────────────────────────────┘
```

---

## Data Source

| Property | Value |
|---|---|
| Provider | Forex Factory |
| Calendar CDN | `https://nfs.faireconomy.media/ff_calendar_thisweek.json` |
| Authentication | None required |
| Cloudflare | Present on `forexfactory.com` — **not** on the CDN |
| Format | JSON array |
| Time zone | US Eastern (EST/EDT) — converted to UTC by normalizer |
| Update cadence | ~5 minutes (CDN `Cache-Control: max-age=300`) |

---

## Request Lifecycle

Every public API request follows this path:

```
1. Frontend sends GET /api/v1/intelligence/calendar
2. FastAPI routes to CalendarEndpoint
3. CalendarEndpoint calls connector_cache.get_calendar("thisweek")
4. MemoryCache returns the pre-populated MFIPEvent list
5. Endpoint serialises to CalendarResponse (MFIPEventOut[])
6. Response returned to frontend

No network call to Forex Factory is ever made during step 1-6.
```

The background path (invisible to the client):

```
APScheduler fires CalendarJob every 5 minutes
  → CDNFetcher.fetch_calendar("thisweek", current_etag)
      → If 304 Not Modified: record success, return (cache unchanged)
      → If 200 OK:
          → CalendarParser.parse_calendar_json(body)
          → SchemaValidator.validate_and_build_events(raw, source_week)
              → TimeNormalizer: ISO 8601 + EST → UTC
              → ImpactNormalizer: "High" → ImpactLevel.HIGH
              → CategoryInferrer: title keywords → EventCategory
              → EventIdBuilder: SHA-256(provider+currency+timestamp+title)
          → DataQuality.check_quality(events)
          → MemoryCache.set_calendar("thisweek", events)
          → HealthReporter.record_success(job_id, response_time_ms)
```

---

## Scheduler Lifecycle

### Startup sequence (FastAPI lifespan)

```python
# src/api/main.py — lifespan()
connector = ForexFactoryConnector()
await connector.startup()           # warm cache + start scheduler
app.state.ff_connector = connector
```

Inside `connector.startup()`:
1. `asyncio.gather(run_calendar_job("thisweek"), run_calendar_job("nextweek"), run_calendar_job("lastweek"))` — parallel warm-up
2. `build_scheduler()` — registers all jobs with APScheduler and HealthReporter
3. `scheduler.start()` — begins polling

### Shutdown sequence

```python
# src/api/main.py — lifespan() teardown
await app.state.ff_connector.shutdown()
```

Inside `connector.shutdown()`:
1. `scheduler.shutdown(wait=False)` — stops accepting new runs; running jobs complete
2. `HTTPClient.close()` — releases the shared httpx AsyncClient

---

## Cache Lifecycle

The cache is a single in-memory `ConnectorCache` instance (module-level singleton).

| Week | TTL | Stale-serve window | Notes |
|---|---|---|---|
| `thisweek` | 5 min | 30 min | Serves stale with `is_stale: true` flag on CDN failure |
| `nextweek` | 10 min | 4 hr | Lower urgency — data changes infrequently |
| `lastweek` | 1 hr | 24 hr | Historical — effectively frozen after week end |

**Conditional GET:** every poll sends `If-None-Match` with the stored ETag. A `304 Not Modified` response costs nearly zero processing — the cache is unchanged.

**Cache Not Ready:** if the connector is still warming up and an API request arrives, endpoints return `HTTP 503 Service Unavailable` with a descriptive message. This should not happen in normal operation because warm-up runs before `yield` in the lifespan.

---

## API Endpoints

Base path: `/api/v1/intelligence`

### Calendar

| Method | Path | Description |
|---|---|---|
| `GET` | `/calendar` | All events for a given week |
| `GET` | `/calendar/high-impact` | HIGH-impact events for a given week |

Query parameters:
- `week` — `thisweek` (default) \| `nextweek` \| `lastweek`
- `currency` — ISO 4217 code (e.g. `USD`, `EUR`)
- `impact` — `HIGH` \| `MEDIUM` \| `LOW` \| `HOLIDAY`

### Events

| Method | Path | Description |
|---|---|---|
| `GET` | `/events/today` | All events for today (UTC date) |
| `GET` | `/events/high-impact` | HIGH-impact events |
| `GET` | `/events/next` | Next single upcoming event |

Query parameters for `/events/next`:
- `high_impact_only=true` — restrict to HIGH-impact events only

### Speeches

| Method | Path | Description |
|---|---|---|
| `GET` | `/speeches` | Central bank speeches, testimonies, and press conferences |

### Economic Intelligence Engine (Phase 3 — implemented)

| Method | Path | Description |
|---|---|---|
| `GET` | `/intelligence/context` | Full context: active events + upcoming + execution scores |
| `GET` | `/intelligence/execution-risk` | Current execution risk score (0–100) |
| `GET` | `/intelligence/readiness` | Current execution readiness score (0–100) |
| `GET` | `/intelligence/active-events` | Released events still carrying influence |
| `GET` | `/intelligence/upcoming-events` | Scheduled events within N hours |
| `GET` | `/intelligence/economic-summary` | Per-currency economic direction summary |

Query parameters for `/upcoming-events`:
- `hours_ahead` — lookahead window in hours (default 24, max 168)
- `currency` — ISO 4217 filter
- `high_impact_only=true` — restrict to HIGH-impact events

### Phase 3 news/sentiment (stubs — 503 pending data source)

| Method | Path | Phase |
|---|---|---|
| `GET` | `/news` | Phase 4 |
| `GET` | `/sentiment` | Phase 4 |

### Health

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Full connector operational dashboard |

---

## Canonical Event Model (MFIPEvent)

Every API response uses the canonical `MFIPEvent` model. No Forex Factory-specific field names are ever exposed.

```json
{
  "event_id":          "a3f72b91c4d5e6f7",
  "provider":          "forex_factory",
  "provider_event_id": "b91c3a2d4e5f6a7b",
  "title":             "US Non-Farm Employment Change",
  "currency":          "USD",
  "country":           "US",
  "timestamp_utc":     "2026-07-04T12:30:00Z",
  "is_all_day":        false,
  "impact":            "HIGH",
  "is_high_impact":    true,
  "is_speech":         false,
  "category":          "EMPLOYMENT",
  "forecast":          "185K",
  "previous":          "177K",
  "actual":            null,
  "status":            "SCHEDULED",
  "last_updated":      "2026-07-04T09:15:32Z",
  "metadata": {
    "source_week": "thisweek",
    "raw_date":    "2026-07-04T08:30:00",
    "raw_time":    "",
    "raw_impact":  "High",
    "url":         ""
  }
}
```

The `metadata` field contains provider-specific raw values for debugging and audit. Consumer code must never depend on metadata fields — they are implementation details.

---

## Health Monitoring

`GET /api/v1/intelligence/health` returns a comprehensive operational snapshot:

```json
{
  "schema_version":             "1.0.0",
  "provider":                   "forex_factory",
  "connector_status":           "ok",
  "scheduler_running":          true,
  "started_at":                 "2026-07-06T07:00:00Z",
  "uptime_s":                   3600.0,
  "cache_populated": {
    "thisweek": true,
    "nextweek": true,
    "lastweek": true
  },
  "calendar_events_total":      87,
  "calendar_events_high_impact": 12,
  "speeches_cached":            4,
  "news_items_cached":          0,
  "jobs": {
    "calendar:thisweek": {
      "status":          "ok",
      "poll_interval_s": 300,
      "last_success":    "2026-07-06T09:55:01Z",
      "last_failure":    null,
      "next_run":        "2026-07-06T10:00:01Z",
      "success_count":   73,
      "failure_count":   0,
      "retry_count":     0,
      "circuit_open":    false,
      "avg_response_ms": 412.5,
      "last_response_ms": 398.0
    }
  },
  "calendar_poll_s":  300,
  "news_poll_s":      120,
  "sentiment_poll_s": 300,
  "speeches_poll_s":  600
}
```

### Status values

| Status | Meaning |
|---|---|
| `ok` | All jobs succeeding, cache fresh |
| `initializing` | Connector just started — not yet polled |
| `degraded` | At least one job failing but cache still serveable |
| `down` | Circuit breaker open — repeated failures, cache is stale |

### Circuit breaker

The circuit breaker trips after `CIRCUIT_BREAKER_THRESHOLD` consecutive failures (default: 5). When open, the job is skipped and the stale cache continues to serve. The breaker resets on the next successful response.

---

## Polling Configuration

All intervals are configurable via environment variables with `FF_` prefix:

| Variable | Default | Description |
|---|---|---|
| `FF_CALENDAR_POLL_SECONDS` | `300` | Calendar poll interval |
| `FF_NEWS_POLL_SECONDS` | `120` | News poll interval (Phase 3) |
| `FF_SENTIMENT_POLL_SECONDS` | `300` | Sentiment poll interval (Phase 3) |
| `FF_SPEECHES_POLL_SECONDS` | `600` | Speeches derived-view interval |
| `FF_HIGH_IMPACT_LOOKAHEAD_MINUTES` | `15` | Adaptive poll lookahead window |
| `FF_HIGH_IMPACT_POLL_SECONDS` | `60` | Adaptive poll interval during HIGH-impact window |
| `FF_CIRCUIT_BREAKER_THRESHOLD` | `5` | Failures before circuit opens |
| `FF_CDN_BASE_URL` | `https://nfs.faireconomy.media` | CDN hostname (update without code change) |

---

## Error Recovery

| Error | Action | Cache |
|---|---|---|
| HTTP 5xx | Retry (max 3, exponential backoff: 5s → 10s → 20s) | Serve stale, mark `is_stale: true` |
| HTTP 304 Not Modified | Record success, no processing | Unchanged |
| HTTP 404 | Log critical, open circuit | Serve stale |
| HTTP 429 Too Many Requests | Honour `Retry-After` header | Serve stale |
| JSON parse error | Log with raw body hash, discard | Unchanged |
| Pydantic validation failure | Log invalid fields, skip bad events, keep valid ones | Partial update |
| Network timeout | Retry with backoff | Serve stale |

---

## Adding a New Provider (Phase 3+)

Because all downstream components speak `MFIPEvent`, adding FXStreet or TradingEconomics requires:

1. Create `fxstreet_connector/fetcher/`, `parser/`, `validator/`, `normalizer/` (same pattern)
2. In the validator, output `MFIPEvent(provider="fxstreet", ...)`
3. Register jobs in the existing `PollScheduler`
4. Mount the new endpoints under `/api/v1/intelligence/`

The cache, health monitor, API schemas, and frontend code require **no changes**.

---

## Economic Intelligence Engine (EIE) — Phase 3 Architecture

The EIE is a fully deterministic rule engine that converts raw `MFIPEvent` objects from the FF Connector cache into structured `EconomicIntelligenceReport` objects.

### Pipeline

```
MFIPEvent (from FF Connector Cache)
    │
    ├─ EventClassifier       → EventType (EMPLOYMENT, INFLATION, INTEREST_RATE…)
    │
    ├─ SurpriseCalculator    → SurpriseResult (raw, %, class, direction)
    │
    ├─ DirectionRuleEngine   → DirectionSignal (BULLISH/BEARISH/NEUTRAL, confidence, rationale)
    │
    ├─ ImpactCalculator      → impact_score (0–100)
    │
    ├─ DecayCalculator       → remaining_influence (0–100), event_age_hours
    │
    └─ ExecutionRiskCalculator → execution_risk (0–100), execution_readiness (0–100)
                                (computed from the full set of events, not per-event)
    │
    └─ EconomicIntelligenceReport (canonical output)
```

### Direction Rules

Directions are centralized in `economic_intelligence/direction_engine/rules_registry.py`. Each rule captures the economic intuition in a single `higher_is_bullish` flag:

| Event Type | Higher is Bullish? | Rationale |
|---|---|---|
| EMPLOYMENT | ✓ | More jobs → strong economy → bullish |
| UNEMPLOYMENT | ✗ | Higher unemployment → weak economy → bearish |
| JOBLESS_CLAIMS | ✗ | More claims → rising layoffs → bearish |
| INFLATION | ✓ | Higher CPI → hawkish CB expectations → bullish |
| INTEREST_RATE | ✓ | Higher rates → capital inflows → bullish |
| GDP | ✓ | Higher output → strong economy → bullish |
| PMI | ✓ | PMI > 50 = expansion → bullish |
| WAGES | ✓ | Higher wages → inflation expectations → bullish |

### Decay Curves

Each event type has a half-life that determines how quickly its influence fades:

| Event Type | Half-life (hours) | Min Influence |
|---|---|---|
| INTEREST_RATE | 24 | 15% |
| CENTRAL_BANK_SPEECH | 12 | 10% |
| INFLATION | 12 | 5% |
| EMPLOYMENT | 8 | 5% |
| GDP | 8 | 3% |
| PMI | 4 | 2% |
| HOUSING | 4 | 1% |

Formula: `remaining = base_score × 0.5^(hours_elapsed / half_life)`

### Execution Risk Scoring

| Factor | Max Contribution | Notes |
|---|---|---|
| Time to next HIGH event | 50 pts | < 5 min = full 50 pts |
| Remaining influence of active events | 30 pts | proportional to influence % |
| Event clustering (≥ 2 events in 30 min) | 20 pts | 7 pts per additional event |

Readiness = `100 − risk`, with a +20 bonus in the post-release momentum window (strong surprise released within last 30 min).

---

## Running the Tests

```bash
# Install dependencies
pip install pytest pytest-asyncio httpx

# Run connector tests only (no database / ML required)
pytest forex_factory_connector/tests/ -v

# Run just the integration tests
pytest forex_factory_connector/tests/test_integration.py -v

# Run EIE unit tests (no network, no ML, no DB)
pytest economic_intelligence/tests/ -v

# Run with coverage
pytest forex_factory_connector/tests/ economic_intelligence/tests/ \
    --cov=forex_factory_connector --cov=economic_intelligence \
    --cov-report=term-missing
```
