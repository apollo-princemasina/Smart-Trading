# Application Backend — Phase 6

## Overview

The Application Backend is the orchestration layer that connects Phases 1–5 to the frontend. It owns APIs, authentication, WebSocket communication, application persistence, health monitoring, request routing, and service coordination. It contains **no trading logic** — that is owned entirely by the Phase 1–5 engines.

Think of this layer as the operating system of MFIP. It provides:

- A single dashboard endpoint that aggregates all engine outputs in one call
- Versioned REST APIs for every application concern
- WebSocket events for live frontend updates
- Persistent storage for decisions, settings, model versions, system logs, and users
- JWT authentication foundation (wired up, enforcement opt-in)
- Request tracing via correlation IDs
- Deep health monitoring across all 7 engine components

---

## Architecture Position

```
Frontend (Next.js)
        │
        ▼
Application Backend  ←── This layer (Phase 6)
        │
        ├── Decision Fusion Engine      (Phase 5)
        ├── Market Intelligence AI      (Phase 4)
        ├── Economic Intelligence       (Phase 3)
        ├── Forex Factory Connector     (Phase 2)
        ├── Rolling Buffer Manager      (Phase 1)
        └── ML Inference Pipeline       (Phase 1)
```

---

## Folder Structure

```
src/
├── api/
│   ├── main.py                     Application entry point + lifespan
│   ├── core/
│   │   ├── config.py               Centralised settings (pydantic BaseSettings)
│   │   ├── dependencies.py         FastAPI Depends() providers for all services
│   │   ├── exceptions.py           Custom exception hierarchy + HTTP handlers
│   │   └── logging.py              Loguru setup (JSON in prod, coloured in dev)
│   ├── v1/
│   │   ├── router.py               Mounts all v1 endpoint routers
│   │   └── endpoints/
│   │       ├── health.py           GET /api/v1/health/* (K8s probes)
│   │       ├── predictions.py      GET /api/v1/predictions/* (ML signals)
│   │       ├── market.py           GET /api/v1/market/* (candles + regime)
│   │       ├── dashboard.py        GET /api/v1/dashboard
│   │       ├── system.py           /api/v1/system/*
│   │       ├── settings_ep.py      /api/v1/settings/*
│   │       ├── models_registry.py  /api/v1/models/*
│   │       ├── history.py          /api/v1/history/*
│   │       └── auth.py             /api/v1/auth/*
│   ├── schemas/
│   │   ├── health.py, prediction.py, market.py  (Phase 1)
│   │   ├── dashboard.py            DashboardResponse
│   │   ├── system.py               SystemHealthResponse, VersionInfo, etc.
│   │   ├── settings.py             SettingOut, SettingUpdateRequest
│   │   ├── models_registry.py      ModelRegistryOut, ModelRegistrationRequest
│   │   ├── history.py              PredictionHistoryItem, DecisionHistoryItem
│   │   └── auth.py                 LoginRequest, TokenResponse, UserOut
│   └── websocket/
│       ├── manager.py              WebSocketManager (enhanced with subscriptions)
│       └── events.py               WSEventType registry
├── auth/
│   ├── jwt_utils.py                Token generation + validation (HS256)
│   └── dependencies.py             get_current_user / get_current_user_optional
├── middleware/
│   └── request_id.py               X-Request-ID + X-Process-Time headers
├── database/
│   ├── base.py                     DeclarativeBase
│   ├── session.py                  Async engine + session factory
│   ├── models/
│   │   ├── prediction.py           Phase 1 (stable)
│   │   ├── outcome.py              Phase 1 (stable)
│   │   ├── model_meta.py           Phase 1 (stable)
│   │   ├── decision_history.py     Persisted DFE outputs
│   │   ├── app_settings.py         Runtime key-value settings
│   │   ├── system_log.py           Structured system event log
│   │   ├── notification_history.py WebSocket broadcast audit log
│   │   ├── user.py                 Auth foundation
│   │   └── model_registry.py       Full model versioning
│   └── repositories/
│       ├── base.py                 BaseRepository[T] (stable)
│       ├── prediction_repo.py      Phase 1 (stable)
│       ├── decision_repo.py        Decision history queries
│       ├── settings_repo.py        Settings CRUD + upsert
│       └── system_log_repo.py      Log queries + write helper
├── services/
│   ├── (Phase 1 services — stable)
│   ├── dashboard_service.py        Single-call aggregation of all engines
│   ├── decision_service.py         DFE + DB persistence wrapper
│   ├── model_registry_service.py   Model version management
│   ├── history_service.py          Unified prediction + decision history
│   ├── settings_service.py         Typed runtime settings CRUD
│   ├── notification_service.py     WebSocket broadcast + log
│   └── system_health_service.py    Deep health across all 7 engines
└── tests/
    ├── conftest.py                 Shared fixtures + in-memory SQLite setup
    ├── test_dashboard.py
    ├── test_system.py
    ├── test_settings.py
    ├── test_models_registry.py
    ├── test_history.py
    ├── test_auth.py
    └── test_websocket.py
```

---

## REST API Reference

All routes are prefixed `/api/v1/`.

### Dashboard

| Method | Path | Description |
|---|---|---|
| GET | `/dashboard` | Full snapshot: decision + prediction + regime + MIA + EIE + buffer + system |

### System

| Method | Path | Description |
|---|---|---|
| GET | `/system/health` | Deep health check across all 7 engines + DB + WebSocket |
| GET | `/system/status` | Quick status — which engines are online, uptime |
| GET | `/system/version` | API version, schema versions, active model |
| GET | `/system/logs` | Recent system log entries. Query: `level`, `component`, `limit` |

### Settings

| Method | Path | Description |
|---|---|---|
| GET | `/settings` | All settings (secrets redacted) grouped by category |
| GET | `/settings/{key}` | Single setting with metadata |
| PUT | `/settings/{key}` | Update a setting at runtime (no restart required) |

### Model Registry

| Method | Path | Description |
|---|---|---|
| GET | `/models` | All registered model versions, newest first |
| GET | `/models/active` | Currently active model with full governance metadata |
| GET | `/models/{id}` | Model by ID |
| POST | `/models/register` | Register new model. Deactivates current active. |

### History

| Method | Path | Description |
|---|---|---|
| GET | `/history/predictions` | Paginated ML prediction history. Filters: `symbol`, `direction` |
| GET | `/history/decisions` | Paginated DFE decision history. Filters: `recommendation`, `strength`, `after`, `before` |
| GET | `/history/combined` | Interleaved predictions + decisions sorted by time |

### Auth

| Method | Path | Description |
|---|---|---|
| POST | `/auth/login` | Email + password → access token + refresh token |
| POST | `/auth/refresh` | Refresh token → new access token |
| GET | `/auth/me` | Current authenticated user profile |

### Phase 1–5 (unchanged)

| Prefix | Owner |
|---|---|
| `/health/*` | Phase 1 — HealthMonitor |
| `/predictions/*` | Phase 1 — PredictionService |
| `/market/*` | Phase 1 — InferenceEngine |
| `/intelligence/*` | Phase 2 — ForexFactoryConnector |
| `/eie/*` | Phase 3 — EconomicIntelligenceEngine |
| `/mia/*` | Phase 4 — MarketIntelligenceAIEngine |
| `/decision/*` | Phase 5 — DecisionFusionEngine |

---

## WebSocket Architecture

**Endpoint:** `ws://host/ws`

### Connection Lifecycle

1. Client connects. Manager calls `accept()`.
2. Server sends `connection_ack` event immediately.
3. Client optionally sends `{"type": "subscribe", "events": [...]}` to filter events.
4. Server sends `subscription_ack` with the confirmed subscription list.
5. Server broadcasts events as they occur.
6. Client sends `{"type": "ping"}` for keepalive; server responds with `pong`.
7. Client disconnects. Manager removes connection and subscription entry.

### Subscription Filtering

Clients that never send a subscribe message receive **all events** (backwards compatible). Clients that subscribe receive only the requested subset. Clients can unsubscribe (return to all events) by sending `{"type": "unsubscribe"}`.

### Event Registry

| Event | Phase | Trigger |
|---|---|---|
| `signal_update` | 1 | New BUY/SELL/HOLD from inference |
| `regime_update` | 1 | Market regime changed |
| `candle_update` | 1 | New M15 candle in buffer |
| `health_update` | 1 | 60s health tick |
| `decision_update` | 5 | New DecisionObject produced |
| `mia_update` | 4 | New MIA output |
| `eie_update` | 3 | EIE active reports changed |
| `system_status` | 6 | Engine state transition |
| `scheduler_tick` | 6 | M15 cron fired |
| `model_loaded` | 6 | New model bundle registered |
| `connection_ack` | 6 | Sent to client on connect |
| `subscription_ack` | 6 | Sent after subscribe message |
| `ping` / `pong` | 6 | Keepalive |

---

## Database Schema

The following tables are new in Phase 6 (existing Phase 1 tables unchanged):

### decision_history

Persists every `DecisionObject` produced by the DFE. Indexed by `decision_id` (unique), `recommendation`, and `generated_at`.

### app_settings

Key-value store for runtime-configurable settings. Values stored as text, typed on read by `SettingsService`. Supports categories: `inference`, `display`, `notifications`, `thresholds`, `general`.

### system_logs

Structured system event log. Fields: `level`, `component`, `event_type`, `message`, `details` (JSON), `correlation_id`. Used by `GET /system/logs`.

### notification_history

Audit log for every WebSocket broadcast. Records event type, payload, delivery count, and any error.

### users

Auth foundation. Stores bcrypt-hashed password, role (`viewer` / `analyst` / `admin`), subscription tier (`free` / `pro` / `enterprise`), and per-user preferences JSON.

### model_registry

Comprehensive model versioning: git commit, feature schema version, label version, decision schema version, pipeline version, per-class metrics (precision/recall/F1 for BUY and SELL). One row per deployed bundle. `is_active=True` marks the current model.

---

## Auth Foundation

Auth is wired up and ready but **enforcement is opt-in** — no existing routes are protected. Future phases protect routes by adding `Depends(get_current_user)`.

**Token flow:**

```
POST /auth/login
  → validates email + bcrypt password
  → returns { access_token (HS256, 30min), refresh_token (7 days) }

POST /auth/refresh
  → validates refresh token
  → returns new access_token + new refresh_token

GET /auth/me
  → requires Authorization: Bearer <access_token>
  → returns UserOut
```

**Dependency injection:**

```python
from src.auth.dependencies import get_current_user, get_current_user_optional

# Required auth — raises 401 if missing/invalid
@router.get("/protected")
async def protected(user=Depends(get_current_user)):
    ...

# Optional auth — returns None if no token
@router.get("/optional")
async def optional(user=Depends(get_current_user_optional)):
    ...

# Role-based
@router.post("/admin")
async def admin_action(user=Depends(require_role("admin"))):
    ...
```

---

## Startup Lifecycle

The Phase 6 services start **after** all Phase 1–5 engines so they can safely read from `app.state`:

```
... Phase 1–5 engines ...
DecisionFusionEngine (Phase 5)

SettingsService          ← loads runtime settings from DB
ModelRegistryService     ← auto-registers current bundle from PipelineManager
DecisionService          ← wraps DFE + decision_history persistence
HistoryService           ← unified paginated history
NotificationService      ← WebSocket broadcast + audit log
SystemHealthService      ← reads from all app.state engines
DashboardService         ← aggregates everything for /dashboard
```

**Shutdown order:** NotificationService flush → DFE → MIA → EIE → FF Connector → Scheduler → Buffer cache save.

---

## Request Lifecycle

```
Client → HTTP Request
  │
  └─ RequestIDMiddleware
        → Generate/read X-Request-ID
        → Start timer
        └─ CORSMiddleware
              └─ FastAPI Route Handler
                    → Depends() injects services from app.state
                    → Service reads from engine or DB
                    → Returns Pydantic model
  ← HTTP Response + X-Request-ID + X-Process-Time headers
```

---

## Health Monitoring

`SystemHealthService.full_health()` checks each component independently and returns:

```json
{
  "status": "operational",
  "uptime_seconds": 3600.0,
  "components": {
    "rolling_buffer":       { "status": "ok", "ready": true },
    "ml_pipeline":          { "status": "ok", "model_loaded": true },
    "scheduler":            { "status": "ok", "running": true },
    "forex_factory":        { "status": "ok" },
    "economic_intelligence": { "status": "ok", "running": true },
    "market_intelligence_ai": { "status": "ok", "running": true },
    "decision_fusion":      { "status": "ok", "running": true },
    "websocket":            { "status": "ok", "active_connections": 3 }
  }
}
```

Component status values: `ok` | `degraded` | `error` | `stopped`.

---

## Configuration

New environment variables added in Phase 6:

| Variable | Default | Description |
|---|---|---|
| `JWT_ACCESS_TOKEN_EXPIRE_MINUTES` | `30` | Access token TTL |
| `JWT_REFRESH_TOKEN_EXPIRE_DAYS` | `7` | Refresh token TTL |
| `AUTH_ENABLED` | `false` | Enable auth enforcement globally |

All existing variables (`SECRET_KEY`, `DATABASE_URL`, etc.) are unchanged.

---

## New Dependencies

Added to `requirements.txt`:

```
python-jose[cryptography]>=3.3.0    # JWT encoding/decoding
passlib[bcrypt]>=1.7.4              # Password hashing
```

Install: `pip install python-jose[cryptography] passlib[bcrypt]`

---

## Testing

```bash
# Run Application Backend tests only
pytest src/tests/ -v

# Run with coverage
pytest src/tests/ --cov=src --cov-report=term-missing

# Run specific module
pytest src/tests/test_auth.py -v
```

Tests use an in-memory SQLite database — no external services required.

Test coverage:

| Module | Tests | Covers |
|---|---|---|
| `test_dashboard.py` | 4 | Dashboard aggregation, null decisions |
| `test_system.py` | 10 | Health, status, version, logs endpoints |
| `test_settings.py` | 6 | CRUD, round-trip, 404 paths |
| `test_models_registry.py` | 8 | Register, list, active, lookup, 404 |
| `test_history.py` | 8 | Prediction/decision/combined history, filters |
| `test_auth.py` | 10 | JWT utils, login, refresh, /me, role |
| `test_websocket.py` | 10 | Connect, broadcast, subscribe, filter, ping |

---

## Railway Deployment Compatibility

The Application Backend is fully compatible with Railway deployment:

1. **Database**: `DATABASE_URL` auto-converts `postgresql://` → `postgresql+asyncpg://`. Railway injects this automatically.
2. **Auth**: `SECRET_KEY` should be set as a Railway secret (not committed to git).
3. **Port**: `APP_PORT=8000` is configurable via env. Railway assigns `$PORT` — pass it as `APP_PORT=$PORT`.
4. **CORS**: `ALLOWED_ORIGINS` accepts comma-separated origins for the production frontend URL.
5. **Buffer cache**: Uses local disk (`data/buffer_cache/`). On Railway with ephemeral storage, set `BUFFER_CACHE_ENABLED=false` and rely on Twelve Data on every restart.
6. **Logging**: Set `APP_ENV=production` for JSON log format compatible with Railway's log aggregation.

```env
# Railway environment
APP_ENV=production
DATABASE_URL=${{Postgres.DATABASE_URL}}
SECRET_KEY=<generated with: openssl rand -hex 32>
ALLOWED_ORIGINS=https://your-frontend.vercel.app
APP_PORT=${{PORT}}
BUFFER_CACHE_ENABLED=false
```
