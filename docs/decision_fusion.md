# Decision Fusion Engine (DFE) — Phase 5

## Overview

The Decision Fusion Engine is the central decision-making component of MFIP. It combines every available intelligence signal into a single, explainable, versioned trading recommendation without calling any external AI or model inference service. All logic is deterministic and fully reproducible.

**What it does:**
- Receives the full intelligence bundle (ML prediction, EIE reports, MIA output, live market state)
- Normalises each source into a common `EvidenceItem` representation
- Measures directional agreement and conflict across all sources
- Computes a blended confidence score using a 7-step methodology
- Applies 13 prioritised business rules (safety checks, bonuses, caps)
- Generates a `BUY / SELL / WAIT` recommendation with strength and expiry
- Builds a human-readable explanation for every decision
- Caches the current decision and exposes it through a REST API

**What it never does:**
- Call Groq, OpenAI, or any other LLM/inference API
- Execute trades or manage orders
- Make random or non-reproducible decisions

---

## Architecture

```
FusionInput
    │
    ▼
EvidenceCollector          → List[EvidenceItem]
    │
    ▼
AgreementCalculator        → AgreementResult
    │
    ▼
ConfidenceFusion           → float (0–100, preliminary)
    │
    ▼
RuleEvaluator              → RuleEvaluation (deltas, caps, forced_wait)
    │
    ▼
RecommendationGenerator    → RecommendationResult
    │
    ▼
ExplanationBuilder         → ExplanationResult
    │
    ▼
DecisionObject             → stored in DecisionCache → served via REST API
```

---

## Module Structure

```
decision_fusion/
├── __init__.py                        # exports DecisionFusionEngine
├── engine.py                          # orchestrator, APScheduler cycle, health
│
├── models/
│   ├── enums.py                       # Recommendation, RecommendationStrength, ConsensusLevel,
│   │                                  # SourceType, EvidenceDirection, MarketBiasEnum
│   ├── evidence.py                    # EvidenceItem, AgreementResult, RecommendationResult
│   └── fusion_input.py                # FusionInput (canonical input bundle)
│
├── schema/
│   └── decision_object.py             # DecisionObject (Pydantic v2, versioned)
│
├── evidence_engine/
│   └── collector.py                   # EvidenceCollector — normalises ML/EIE/MIA
│
├── agreement_engine/
│   └── calculator.py                  # AgreementCalculator — pairwise weighted agreement
│
├── confidence_engine/
│   └── fusion.py                      # ConfidenceFusion — 7-step methodology
│
├── rule_engine/
│   ├── rules.py                       # 13 BusinessRules (R001–R013)
│   └── evaluator.py                   # RuleEvaluator — EvaluationContext builder + rule runner
│
├── recommendation_engine/
│   └── generator.py                   # RecommendationGenerator — dominant direction → strength
│
├── explanation_engine/
│   └── builder.py                     # ExplanationBuilder — 5 text lists
│
├── recommendation_cache/
│   └── cache.py                       # DecisionCache + module-level singleton
│
├── api/
│   ├── router.py                      # dfe_router (prefix /decision)
│   ├── schemas.py                     # DecisionOut, DecisionResponse, health/confidence/agreement
│   └── endpoints/
│       ├── decision.py                # GET /decision/current, GET /decision/history
│       ├── confidence.py              # GET /decision/confidence
│       ├── agreement.py               # GET /decision/agreement
│       └── health.py                  # GET /decision/health
│
├── utils/
│   ├── config.py                      # DFEConfig — env-driven thresholds
│   ├── logger.py                      # bound loguru logger
│   └── metrics.py                     # DFEMetrics — rolling async deque
│
└── tests/
    ├── conftest.py                    # fixtures + factory helpers
    ├── test_evidence_engine.py
    ├── test_agreement_engine.py
    ├── test_confidence_engine.py
    ├── test_rule_engine.py
    ├── test_recommendation_engine.py
    ├── test_explanation_engine.py
    ├── test_cache.py
    ├── test_api.py
    └── test_engine.py
```

---

## FusionInput

`FusionInput` is the canonical input bundle passed to `DecisionFusionEngine.process()`.

| Field | Type | Source |
|---|---|---|
| `ml_prediction` | `Optional[dict]` | `InferenceEngine.latest_result()` |
| `eie_reports` | `list` | `EconomicIntelligenceEngine.get_active_reports()` |
| `mia_output` | `Optional[Any]` | `MarketIntelligenceAIEngine.latest_output()` |
| `latest_close` | `Optional[float]` | `RollingBufferManager.latest_close()` |
| `buffer_ready` | `bool` | `RollingBufferManager.is_ready()` |
| `buffer_status` | `dict` | `RollingBufferManager.status()` |
| `current_time` | `datetime` | caller-provided UTC timestamp |
| `execution_context` | `Optional[dict]` | reserved — future broker context |
| `cross_asset_intel` | `Optional[dict]` | reserved — future DXY/XAUUSD correlation |
| `cot_intel` | `Optional[dict]` | reserved — future COT positioning data |
| `macro_intel` | `Optional[dict]` | reserved — future macro indicators |

Computed properties (derived from the raw fields without further IO):

- `ml_direction` — "BUY" / "SELL" / "HOLD" / None
- `ml_confidence` — float 0–1 or None
- `ml_regime`, `ml_session` — strings from ML prediction dict
- `eie_execution_risk` — max `execution_risk` across active EIE reports
- `eie_execution_readiness` — min `execution_readiness` across active EIE reports
- `mia_bias` — `MarketBias` enum value or None
- `mia_risk_level` — `RiskLevel` enum value or None
- `mia_confidence` — float 0–1 or None

---

## Evidence-Based Decision Making

### EvidenceItem

All source signals are normalised into a frozen `EvidenceItem` dataclass before any fusion logic runs. This ensures every engine operates on a common, immutable representation regardless of the originating module.

```python
@dataclass(frozen=True)
class EvidenceItem:
    source:      SourceType
    direction:   EvidenceDirection   # BULLISH | BEARISH | NEUTRAL | UNCERTAIN | ABSENT
    confidence:  float               # 0–1
    reliability: float               # source-level weight (ML=0.85, EIE=0.75, MIA=0.70)
    importance:  float               # 0–1, scaled from source signal strength
    timestamp:   datetime
    label:       str                 # human-readable source description
    raw_value:   Optional[float]     # original numeric value if applicable
    metadata:    dict

    @property
    def weight(self) -> float:
        return self.reliability * self.importance * self.confidence

    @property
    def directional_weight(self) -> float:
        # +weight for BULLISH, -weight for BEARISH, 0 for all others
```

### Evidence Collection

`EvidenceCollector.collect(FusionInput) → List[EvidenceItem]` produces one item per source:

**ML Evidence** (`TECHNICAL_ML`)
- Direction: BUY→BULLISH, SELL→BEARISH, HOLD→NEUTRAL
- Confidence: `ml_prediction["confidence"]`
- Reliability: 0.85 (configurable via `ML_RELIABILITY` env var)
- Absent when `ml_prediction` is None

**EIE Evidence** (`FUNDAMENTAL_EIE`)
- Filters reports with `remaining_influence >= threshold` (default: 10.0)
- Aggregates into a single item: weighted average direction via `impact_score × remaining_influence`
- Direction: positive aggregate→BULLISH, negative→BEARISH, near-zero→NEUTRAL
- Reliability: 0.75
- Absent when no active reports meet the threshold

**MIA Evidence** (`AI_INTELLIGENCE`)
- Direction: maps `MarketBias` enum to `EvidenceDirection`
- Excluded when `mia_output.is_fallback is True` (fallback outputs produce ABSENT)
- Reliability: 0.70
- Absent when `mia_output` is None or is a fallback

---

## Agreement Engine

The `AgreementCalculator` computes pairwise directional agreement across all non-ABSENT, non-UNCERTAIN items.

**Pair weight** = `(reliability_i × importance_i) × (reliability_j × importance_j)`

For each pair:
- BULLISH ↔ BULLISH, BEARISH ↔ BEARISH → **agreement weight**
- BULLISH ↔ BEARISH → **conflict weight**
- NEUTRAL ↔ anything → **neither** (neutral is informative but not directionally aligned)

```
agreement_score = (total_agreement_weight / total_pair_weight) × 100
conflict_score  = (total_conflict_weight  / total_pair_weight) × 100
```

`ConsensusLevel` is derived from the `agreement_score`:

| Score | Consensus |
|---|---|
| < 40 | WEAK |
| 40–60 | MODERATE |
| 60–80 | STRONG |
| > 80 | VERY_STRONG |

---

## Confidence Methodology (7 Steps)

`ConfidenceFusion.compute()` builds the preliminary confidence score:

1. **Base**: ML confidence × 100. Falls back to MIA confidence × 100 if no ML. Final fallback: 50.
2. **EIE alignment bonus**: +10 if EIE direction matches the dominant direction.
3. **MIA alignment bonus**: +8 if MIA direction matches the dominant direction.
4. **Triple confirmation bonus**: +10 if all three sources agree.
5. **Consensus multiplier**: VERY_STRONG ×1.10 / STRONG ×1.00 / MODERATE ×0.90 / WEAK ×0.80.
6. **Conflict penalty**: conflict > 70% → −20 pts; conflict > 50% → −10 pts.
7. **Execution risk penalty**: EIE exec_risk > 80% → −15 pts; exec_risk > 50% → −8 pts.
   AI risk CRITICAL → −15 pts.
8. **Clip** to [0, 100].

---

## Business Rules

`RuleEvaluator` evaluates 13 `BusinessRule` instances in ascending priority order. `FORCE_WAIT` rules set `stops_evaluation=True`, ending rule evaluation immediately.

| Rule | Priority | Trigger | Action |
|---|---|---|---|
| R001 | 1 | No evidence at all | FORCE_WAIT |
| R002 | 2 | Buffer not ready | FORCE_WAIT |
| R003 | 5 | conflict > 70 AND confidence < 40 | FORCE_WAIT |
| R004 | 8 | ML direction = HOLD | FORCE_WAIT |
| R005 | 10 | EIE execution_risk ≥ 80% | REDUCE_CONFIDENCE |
| R006 | 15 | ML and EIE directly oppose each other | REDUCE_CONFIDENCE |
| R007 | 20 | conflict > 50 with ≥ 2 sources | REDUCE_CONFIDENCE |
| R008 | 25 | MIA risk_level = CRITICAL | REDUCE_CONFIDENCE |
| R009 | 30 | Triple BUY confirmation | BOOST_CONFIDENCE |
| R010 | 31 | Triple SELL confirmation | BOOST_CONFIDENCE |
| R011 | 35 | Only one source available | CAP_STRENGTH = MODERATE |
| R012 | 40 | VERY_STRONG consensus + ≥ 2 sources | BOOST_CONFIDENCE +5 |
| R013 | 100 | Final confidence < 30 | FORCE_WAIT |

`RuleEvaluation` carries: `forced_wait`, `forced_wait_reason`, `confidence_delta`, `strength_cap`, `triggered_rules`, `rule_reasons`.

---

## Recommendation Engine

`RecommendationGenerator.generate()` determines the final recommendation from the adjusted confidence:

1. **Dominant direction**: weighted vote over all directional items; BULLISH→BUY, BEARISH→SELL, else WAIT.
2. **Apply rule delta**: `adjusted_confidence = preliminary_confidence + confidence_delta`.
3. **Determine strength** from adjusted confidence:
   - < 40 → WEAK
   - 40–60 → MODERATE
   - 60–80 → STRONG
   - > 80 → VERY_STRONG
4. **Consensus cap**: WEAK consensus → max strength = MODERATE.
5. **Strength cap from R011**: single-source cap at MODERATE.
6. `RecommendationStrength` ordering: WEAK < MODERATE < STRONG < VERY_STRONG.

**Alignment signals** (stored in DecisionObject):

- `technical_alignment`: signed ML confidence — BUY → +confidence, SELL → −confidence, HOLD → 0.
- `fundamental_alignment`: weighted average across EIE reports by `impact_score × remaining_influence`.

**Decision expiry** (time until cached decision is stale):

| Recommendation + Strength | Expires after |
|---|---|
| WAIT | 5 min |
| WEAK | 15 min |
| MODERATE | 30 min |
| STRONG | 60 min |
| VERY_STRONG | 120 min |

---

## Explanation Engine

`ExplanationBuilder.build()` returns an `ExplanationResult` with five string lists used verbatim in `DecisionObject`:

| Field | Content |
|---|---|
| `primary_reasons` | Top 3 supporting evidence items + forced_wait reason (if any) |
| `supporting_evidence` | All items aligned with recommendation direction + agreement source names |
| `conflicting_reasons` | Items opposing the recommendation direction |
| `confidence_drivers` | ML anchor, consensus level, rule adjustments, final score |
| `risk_factors` | Execution warnings, AI risk level, contradicting-bias flag, penalty reasons |

---

## Decision Object

`DecisionObject` is a Pydantic v2 model. It is the canonical output of the DFE and the object stored in cache and returned by the API.

```python
class DecisionObject(BaseModel):
    decision_schema_version: str  = "decision_fusion_v1"
    decision_id:             str  # UUID4, auto-generated
    recommendation:          Recommendation          # BUY | SELL | WAIT
    recommendation_strength: RecommendationStrength  # WEAK | MODERATE | STRONG | VERY_STRONG
    decision_confidence:     float  # 0–100
    agreement_score:         float  # 0–100
    conflict_score:          float  # 0–100
    consensus_level:         ConsensusLevel
    technical_alignment:     float  # −1 to +1
    fundamental_alignment:   float  # −1 to +1
    market_bias:             MarketBiasEnum
    primary_reasons:         List[str]
    supporting_evidence:     List[str]
    conflicting_reasons:     List[str]
    confidence_drivers:      List[str]
    risk_factors:            List[str]
    generated_at:            datetime
    expires_at:              datetime
    has_ml:                  bool
    has_eie:                 bool
    has_mia:                 bool
```

### Schema Versioning

`decision_schema_version = "decision_fusion_v1"` is a hard constant embedded in every `DecisionObject`. Downstream consumers can rely on this field to detect breaking changes. When the schema changes in a backwards-incompatible way, the version is bumped (e.g., `decision_fusion_v2`) and the old schema remains importable for migration purposes.

---

## Decision Cache

`DecisionCache` is a module-level singleton (`decision_cache`) with an `asyncio.Lock` for thread safety.

```python
async def store(decision: DecisionObject) → None   # promotes current → previous
async def invalidate() → None                      # force-expire current
current:  Optional[DecisionObject]                 # property
previous: Optional[DecisionObject]                 # property
get_history(limit: int) → list[DecisionObject]     # newest first
is_expired() → bool
age_seconds() → Optional[float]
seconds_until_expiry() → Optional[float]
stats() → dict
```

History is stored in a `deque(maxlen=DFE_HISTORY_MAX_SIZE)` (default: 100).

---

## Background Cycle

`DecisionFusionEngine.startup()` starts an `APScheduler AsyncIOScheduler` that fires `_cycle()` every `DFE_CYCLE_SECONDS` (default: 60 s).

`_cycle()` logic:
1. Skip if no `_last_input` is cached (engine hasn't processed anything yet).
2. Skip if the current decision is still valid (not expired).
3. Reprocess `_last_input` and store the new decision.

This means decisions refresh automatically when they expire, without requiring a caller to trigger `process()` manually.

---

## Public API

All endpoints are mounted under `/api/v1/decision` via `dfe_router`.

### GET /api/v1/decision/current

Returns the most recent `DecisionObject` from cache.

```json
{
  "decision": {
    "decision_schema_version": "decision_fusion_v1",
    "decision_id": "...",
    "recommendation": "BUY",
    "recommendation_strength": "STRONG",
    "decision_confidence": 72.4,
    "agreement_score": 80.0,
    "conflict_score": 10.0,
    "consensus_level": "STRONG",
    "technical_alignment": 0.72,
    "fundamental_alignment": 0.60,
    "market_bias": "BULLISH",
    "primary_reasons": ["..."],
    "supporting_evidence": ["..."],
    "conflicting_reasons": [],
    "confidence_drivers": ["..."],
    "risk_factors": [],
    "generated_at": "2026-07-06T...",
    "expires_at": "2026-07-06T...",
    "has_ml": true,
    "has_eie": true,
    "has_mia": true
  },
  "is_expired": false,
  "age_seconds": 45.2,
  "seconds_until_expiry": 3554.8
}
```

`decision` is `null` when no decision has been produced yet.

### GET /api/v1/decision/history?limit=20

Returns a list of recent decisions (newest first, default limit 20).

### GET /api/v1/decision/confidence

Returns the confidence breakdown for the current decision.

```json
{
  "has_current_decision": true,
  "decision_confidence": 72.4,
  "recommendation": "BUY",
  "confidence_drivers": ["ML anchor: BULLISH 72%", ...],
  "risk_factors": [],
  "is_expired": false
}
```

### GET /api/v1/decision/agreement

Returns the agreement and conflict scores.

```json
{
  "has_current_decision": true,
  "agreement_score": 80.0,
  "conflict_score": 10.0,
  "consensus_level": "STRONG",
  "aligned_sources": ["TECHNICAL_ML", "AI_INTELLIGENCE"],
  "conflicting_sources": [],
  "is_expired": false
}
```

### GET /api/v1/decision/health

Returns the engine operational status.

```json
{
  "status": "operational",
  "running": true,
  "schema_version": "decision_fusion_v1",
  "current_recommendation": "BUY",
  "recommendation_strength": "STRONG",
  "recommendation_age_s": 45.2,
  "time_until_expiry_s": 3554.8,
  "is_expired": false,
  "agreement_score": 80.0,
  "conflict_score": 10.0,
  "decision_confidence": 72.4,
  "avg_processing_ms": 4.5,
  "total_decisions": 7,
  "cache_size": 7
}
```

Returns `{"status": "offline", "running": false}` when the DFE is not mounted in `app.state`.

---

## Configuration

All thresholds are configurable via environment variables. Defaults are in `decision_fusion/utils/config.py`.

| Variable | Default | Description |
|---|---|---|
| `ML_RELIABILITY` | 0.85 | Source reliability weight for ML predictions |
| `EIE_RELIABILITY` | 0.75 | Source reliability weight for EIE reports |
| `MIA_RELIABILITY` | 0.70 | Source reliability weight for MIA output |
| `CONFIDENCE_MODERATE_THRESHOLD` | 40.0 | Min confidence for MODERATE strength |
| `CONFIDENCE_STRONG_THRESHOLD` | 60.0 | Min confidence for STRONG strength |
| `CONFIDENCE_VERY_STRONG_THRESHOLD` | 80.0 | Min confidence for VERY_STRONG strength |
| `DFE_CYCLE_SECONDS` | 60 | Background reprocessing interval |
| `DFE_HISTORY_MAX_SIZE` | 100 | Rolling history deque size |
| `DFE_EXPIRY_WAIT_S` | 300 | WAIT decision expiry (5 min) |
| `DFE_EXPIRY_WEAK_S` | 900 | WEAK decision expiry (15 min) |
| `DFE_EXPIRY_MODERATE_S` | 1800 | MODERATE decision expiry (30 min) |
| `DFE_EXPIRY_STRONG_S` | 3600 | STRONG decision expiry (60 min) |
| `DFE_EXPIRY_VERY_STRONG_S` | 7200 | VERY_STRONG decision expiry (120 min) |

---

## Testing

The test suite covers all six pipeline engines, the cache, the API endpoints, and end-to-end engine flows.

```bash
pytest decision_fusion/tests/ -v
```

Key test categories:

| File | Tests | Coverage |
|---|---|---|
| `test_evidence_engine.py` | 17 | ML/EIE/MIA collection, ABSENT items, fallback exclusion |
| `test_agreement_engine.py` | 11 | Pairwise agreement, conflict, NEUTRAL handling |
| `test_confidence_engine.py` | 9 | 7-step methodology, bonuses, penalties, clipping |
| `test_rule_engine.py` | 13 | All 13 rules, FORCE_WAIT stops evaluation |
| `test_recommendation_engine.py` | 14 | Dominant direction, strength, caps, alignment |
| `test_explanation_engine.py` | 8 | All 5 text lists, risk factors |
| `test_cache.py` | 13 | Store/promote, expiry, history, invalidate, stats |
| `test_api.py` | 18 | All 5 endpoints, with-decision and empty states |
| `test_engine.py` | 11 | End-to-end: BUY/SELL/WAIT scenarios, determinism |

---

## Design Principles

- **Deterministic**: given the same `FusionInput` and `current_time`, the same `DecisionObject` is always produced.
- **Explainable**: every recommendation carries five text lists documenting why it was made.
- **Versioned**: `decision_schema_version` is embedded in every output object.
- **Decoupled**: `FusionInput` uses `Optional[Any]` for EIE/MIA fields; all access is via `getattr()` so the DFE has no import dependency on upstream modules.
- **Testable**: each engine is independently testable; the full pipeline is tested end-to-end.
- **Safe by default**: a fallback `DecisionObject` with `recommendation=WAIT` is returned on any internal error; the application never crashes due to a DFE failure.

---

## Future Extensibility

Reserved `FusionInput` fields provide named extension points for planned future phases:

| Field | Intended source |
|---|---|
| `execution_context` | Phase 6 — broker position and margin state |
| `cross_asset_intel` | Phase 7 — DXY / XAUUSD correlation engine |
| `cot_intel` | Phase 8 — COT report positioning data |
| `macro_intel` | Phase 9 — macroeconomic indicator feeds |

Adding a new source requires: (1) populating the reserved field in `FusionInput`, (2) adding an evidence collector method in `EvidenceCollector`, (3) adding a `SourceType` enum value, and (4) adding any new business rules to `rule_engine/rules.py`.
