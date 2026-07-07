"""
Forex Factory Connector — MFIP Market Intelligence Layer

Phase 1 (investigation) complete. Phase 2 implements the CDN calendar pipeline.
Phase 3 adds news and sentiment scraping (Cloudflare strategy TBD).

Entry points:
    connector.api.router       — FastAPI router to mount on the main app
    connector.scheduler        — build_scheduler() for APScheduler integration
    connector.cache            — connector_cache singleton for direct cache reads
"""
