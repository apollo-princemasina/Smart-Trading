# Market Intelligence AI Layer (MIA) — Architecture Documentation

## Overview

The Market Intelligence AI Layer is Phase 4 of the MFIP intelligence pipeline. It transforms structured economic events and market headlines into institutional-grade market intelligence using a single autonomous AI agent powered by Groq.

The AI behaves like a team of institutional market experts while remaining a single agent with a single inference call per analysis.

**Pipeline position:**
```
Forex Factory Connector
    → Economic Intelligence Engine (EIE)
        → Market Context Compiler (MIA)        ← assembles context
            → Market Intelligence Agent (MIA)  ← single AI inference
                → Schema Validator
                    → AI Cache
                        → Execution Context Engine (future)
```

---

## Why Context Engineering Replaces Prompt Engineering

The previous approach used multiple specialised prompts — one per analyst type (HeadlineAnalyst, EventInterpreter, SpeechAnalyst, ContradictionDetector, NarrativeEngine). This approach was rejected because:

| Prompt Engineering (rejected)           | Context Engineering (current)            |
|-----------------------------------------|------------------------------------------|
| One prompt per event type (CPI, NFP…)  | One permanent system prompt, any input   |
| Manual routing decides which prompt     | AI reasons autonomously from context     |
| Adding event types requires new prompts | New context fields are additive          |
| Prompt maintenance is a bottleneck      | Context assembly is composable           |
| Different analysts → inconsistent schemas | Single schema → consistent contracts   |

**The key insight:** The intelligence should come from the quality of the context, not the sophistication of prompt routing. Give the AI rich, structured, deterministic context — then let it reason from institutional perspectives.

---

## Architecture

```
EIE Intelligence Cache
        │
        ▼
Market Context Compiler ─────────────────────────────────────────────┐
(assembles all available market information)                         │
        │                                                            │
        ▼                                                            ▼
  ContextPayload                                        Single system prompt
  (rich structured context)                             (permanent role + 5 perspectives)
        │                                                            │
        └────────────────────────────────────┬────────────────────────┘
                                             ▼
                              Market Intelligence Agent
                              (single autonomous analyst)
                                             │
                                             ▼
                                         AIGateway
                              (retry, circuit breaker, metrics)
                                             │
                                             ▼
                               Groq (or any provider via ABC)
                                             │
                                             ▼
                                  Schema Validator
                               (JSON + Pydantic validation)
                                             │
                                             ▼
                                          AICache
                              (deterministic TTL cache)
                                             │
                                             ▼
                           MarketIntelligenceOutput
                           (structured JSON contract)
```

---

## Module Structure

```
market_intelligence_ai/
├── __init__.py                        Lazy import of MarketIntelligenceAIEngine
├── engine.py                          Lifecycle orchestrator + APScheduler cycle
│
├── schema/
│   ├── system_prompt.py               Single permanent system prompt (role + 5 perspectives)
│   └── market_intelligence.py         MarketIntelligenceOutput Pydantic model (the output contract)
│
├── market_context_compiler/
│   ├── context_models.py              ContextPayload, EventTrigger, HeadlineTrigger, EIESnapshot
│   └── compiler.py                    MarketContextCompiler — assembles, formats, and keys context
│
├── agent/
│   └── market_agent.py                MarketIntelligenceAgent — the single AI analyst
│
├── providers/
│   ├── base.py                        MarketIntelligenceProvider ABC + error hierarchy
│   ├── groq_provider.py               GroqProvider (default)
│   └── mock_provider.py               MockProvider (testing)
│
├── ai_gateway/
│   ├── circuit_breaker.py             CircuitBreaker (CLOSED→OPEN→HALF_OPEN)
│   └── gateway.py                     AIGateway (retry, validation, metrics)
│
├── response_validator/
│   └── validator.py                   ResponseValidator (JSON extraction, Pydantic validation)
│
├── ai_cache/
│   └── cache.py                       AICache (TTL-based, deterministic keys)
│
├── models/
│   └── enums.py                       MarketBias, Importance, TimeHorizon, RiskLevel, AnalysisType
│
├── utils/
│   ├── config.py                      MIAConfig (env-overridable settings)
│   ├── logger.py                      Bound loguru logger
│   └── metrics.py                     GatewayMetrics (rolling window)
│
├── api/
│   ├── router.py                      mia_router (prefix=/intelligence)
│   ├── schemas.py                     API request/response Pydantic models
│   └── endpoints/
│       ├── analysis.py                POST /analyse/event, POST /analyse/headline, GET /analyses
│       └── health.py                  GET /ai-health
│
└── tests/
    ├── conftest.py                    Shared fixtures (MockProvider, compiler, agent)
    ├── test_context_builder.py        MarketContextCompiler and cache key tests
    ├── test_schema_validation.py      Schema validation and ResponseValidator tests
    ├── test_cache.py                  AICache TTL and concurrency tests
    ├── test_provider.py               MockProvider and provider abstraction tests
    ├── test_gateway.py                AIGateway retry, circuit breaker, fallback tests
    ├── test_agent.py                  MarketIntelligenceAgent end-to-end tests
    └── test_api.py                    HTTP endpoint contract tests
```

---

## The Market Context Compiler

The `MarketContextCompiler` is the heart of the MIA layer. It automatically assembles everything the AI needs into a single rich `ContextPayload`.

### Possible Inputs

| Category | Fields |
|---|---|
| **Economic Event** | title, currency, importance, forecast, previous, actual, surprise class/direction, EIE economic direction |
| **Market Headline** | headline text, source, timestamp, affected currencies |
| **EIE Intelligence** | dominant directions per currency, active events, upcoming high-impact events |
| **Market Session** | ASIA / LONDON / NEW_YORK / OVERLAP / OFF_MARKET (auto-detected from UTC hour) |
| **Metadata** | analysis timestamp, context schema version |
| **Reserved (future)** | technical context, execution context, macro context |

### How It Works

```python
compiler = MarketContextCompiler()

# For an economic event:
payload = compiler.build_for_event(event_trigger, eie_snapshot)

# For a market headline:
payload = compiler.build_for_headline(headline_trigger, eie_snapshot)

# Format as the AI user message:
user_message = compiler.format_as_user_message(payload)

# Deterministic cache key (includes provider_version):
key = MarketContextCompiler.cache_key(payload, provider_version="groq_v1")
```

The compiler formats all available context into a structured user message. The AI receives this alongside the permanent system prompt and reasons from it — no manual event-type routing occurs.

---

## Internal Reasoning Model

The system prompt instructs the AI to reason from **five institutional perspectives** before producing its final structured output. These are **internal reasoning steps** — they are never exposed in the output.

### The Five Perspectives

**1. Economist**
> "What changed economically?"

Evaluates: What economic variable moved (inflation, employment, GDP, interest rates, growth)? How significant is the change relative to expectation and prior trend?

**2. FX Strategist**
> "Which currencies are affected and in which direction?"

Determines: Bullish, bearish, or neutral bias. Expected duration and strength. Cross-currency impacts and existing macro positioning.

**3. Market Microstructure Analyst**
> "How is the market likely to react in the short term?"

Considers: Liquidity conditions, current trading session, event importance, current volatility regime, and immediate market relevance.

**4. Risk Manager**
> "Does this information increase or decrease trading risk?"

Determines: Execution warnings, contradictions with existing data, uncertainty factors, and any confidence adjustments. Assigns `risk_level`: LOW / MEDIUM / HIGH / CRITICAL.

**5. Communicator**
> "Summarise everything clearly."

Produces: A concise 2–3 sentence institutional explanation suitable for display in a professional trading intelligence platform. Populates `market_summary`.

### Why Internal Perspectives?

The AI does not make five separate API calls. All five perspectives are instructed in the single permanent system prompt. The AI synthesises them internally into a single structured JSON response. This produces richer, more consistent analysis than a flat prompt while maintaining a single inference call.

---

## Structured Output Contract

Every analysis call returns exactly this schema (`MarketIntelligenceOutput`):

```json
{
  "analysis_schema_version": "market_intelligence_ai_v1",
  "context_schema_version":  "context_v1",
  "provider":                "groq:llama-3.3-70b-versatile",

  "market_bias":               "BULLISH",
  "affected_currencies":       ["USD"],
  "importance":                "HIGH",
  "confidence":                0.82,
  "expected_duration":         "SHORT_TERM",
  "supports_existing_bias":    true,
  "contradicts_existing_bias": false,
  "risk_level":                "LOW",
  "execution_warning":         null,
  "market_summary":            "2-3 sentence institutional assessment.",

  "timestamp":   "2026-07-01T13:30:00Z",
  "latency_ms":  320.5,
  "is_fallback": false,
  "cache_hit":   false
}
```

This schema is a **versioned contract**. Downstream systems (frontend, future Execution Context Engine) always receive this structure regardless of what triggered the analysis.

---

## Schema Versioning

| Field | Current Value | Bump When |
|---|---|---|
| `analysis_schema_version` | `market_intelligence_ai_v1` | Output fields added, renamed, or semantics changed |
| `context_schema_version`  | `context_v1` | Context fields added, renamed, or semantics changed |
| `provider` | `groq:llama-3.3-70b-versatile` | Provider or model changes |

When `context_schema_version` changes, all cache entries are automatically invalidated because both versions are embedded in the cache key.

---

## Cache

### Key Design

Cache keys are **deterministic** and include all version identifiers:

```python
key = sha256(
    context_schema_version  |   # "context_v1"
    analysis_schema_version |   # "market_intelligence_ai_v1"
    provider_version        |   # "groq_v1"
    analysis_type           |   # "event" | "headline"
    primary_currency        |   # "USD"
    event_id (if event)     |   # e.g. "NFP_2026_01"
    surprise_class (if event)|  # e.g. "LARGE"
    headline_hash (if headline) # sha256[:12] of headline text
)[:32]
```

Repeated identical contexts never trigger unnecessary AI requests. Version bumps automatically invalidate stale entries.

### TTLs (configurable via env)

| Analysis Type | Default TTL |
|---|---|
| Economic event | 30 minutes (`MIA_EVENT_CACHE_TTL_S`) |
| Market headline | 60 minutes (`MIA_HEADLINE_CACHE_TTL_S`) |
| Combined | 10 minutes (`MIA_COMBINED_CACHE_TTL_S`) |

Fallback responses are **never cached**.

---

## Resilience

### Circuit Breaker

```
CLOSED ──(5 failures)──▶ OPEN ──(60s timeout)──▶ HALF_OPEN ──(success)──▶ CLOSED
```

When the circuit is OPEN, all calls return a fallback immediately without touching the provider.

### Retry and Repair

On `ValidationFailure` (AI returned invalid JSON or wrong schema):
1. Attempt 1: original context message
2. Attempt 2: repair prompt (includes the validation error)
3. Attempt 3: repair prompt (second retry)

After 3 failed attempts → fallback response with `is_fallback=True`.

Backoff delays: [0.5s, 2.0s, 5.0s]

---

## Provider Abstraction

```python
class MarketIntelligenceProvider(ABC):
    async def complete(system_prompt, user_prompt, model, temperature, max_tokens) -> ProviderResponse
    async def health_check() -> ProviderHealth
    provider_name: str
    provider_version: str
    default_model: str
    is_configured: bool
```

Swapping providers (Groq → OpenAI, Claude, Gemini, local LLM) requires:
1. Create a new class implementing `MarketIntelligenceProvider`
2. Pass it to `AIGateway(new_provider)`

No downstream code changes. Agents, validators, caches, and API endpoints are provider-agnostic.

---

## API Endpoints

All endpoints are mounted under `/api/v1/intelligence/`:

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/intelligence/analyse/event` | Analyse a released economic event |
| `POST` | `/intelligence/analyse/headline` | Analyse a market headline |
| `GET`  | `/intelligence/analyses` | List recent AI analyses |
| `GET`  | `/intelligence/ai-health` | AI subsystem health dashboard |

---

## Request Lifecycle

```
POST /intelligence/analyse/event
        │
        ▼
  EventTrigger (from request body)
        │
        ▼
  MarketContextCompiler.build_for_event()
  ├── Attaches EIE snapshot (dominant directions, active events, upcoming events)
  ├── Detects current market session (ASIA/LONDON/NEW_YORK/OVERLAP)
  └── Produces ContextPayload
        │
        ▼
  MarketContextCompiler.cache_key(payload)
        │
        ├── Cache HIT → return cached MarketIntelligenceOutput (cache_hit=True)
        │
        └── Cache MISS → MarketIntelligenceAgent.analyze()
                │
                ├── format_as_user_message(payload) → structured user message
                │
                ├── AIGateway.complete(SYSTEM_PROMPT, user_message, ...)
                │   ├── Circuit OPEN? → return fallback immediately
                │   ├── Call Groq (or configured provider)
                │   ├── ResponseValidator.validate() → MarketIntelligenceOutput
                │   └── On ValidationFailure → repair prompt + retry (max 2)
                │
                ├── Cache result (if not fallback)
                │
                └── Return MarketIntelligenceOutput
```

---

## Configuration

| Variable | Default | Description |
|---|---|---|
| `GROQ_API_KEY` | `""` | Groq API key — required for live analysis |
| `MIA_ANALYSIS_MODEL` | `llama-3.3-70b-versatile` | Groq model |
| `MIA_TEMPERATURE` | `0.1` | LLM temperature (low = deterministic) |
| `MIA_MAX_TOKENS` | `1024` | Max tokens per completion |
| `MIA_MAX_RETRIES` | `2` | Retry attempts on validation failure |
| `MIA_CIRCUIT_BREAKER_THRESHOLD` | `5` | Failures before circuit opens |
| `MIA_CIRCUIT_RESET_SECONDS` | `60` | Seconds before OPEN→HALF_OPEN |
| `MIA_EVENT_CACHE_TTL_S` | `1800` | Event analysis cache TTL |
| `MIA_HEADLINE_CACHE_TTL_S` | `3600` | Headline analysis cache TTL |
| `MIA_CYCLE_SECONDS` | `300` | Background auto-analysis cycle interval |

---

## Running Tests

```bash
pytest market_intelligence_ai/tests/ -v
```

No Groq API key required — all tests use `MockProvider`.

---

## Important Constraints

1. The AI **never** produces BUY, SELL, HOLD, LONG, SHORT, ENTER, or EXIT
2. The AI **never** produces specific price levels
3. All AI output passes through `ResponseValidator` before reaching the backend
4. Fallback responses are **never cached**
5. `MarketIntelligenceOutput` is a **versioned contract** — changes require a version bump
6. EIE deterministic calculations are **never overridden** by AI output
7. There is exactly **one AI inference call** per analysis request
8. There is exactly **one system prompt** — it never changes for different event types
