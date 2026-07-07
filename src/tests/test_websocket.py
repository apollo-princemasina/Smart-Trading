"""Tests for the enhanced WebSocket manager."""
from __future__ import annotations

import asyncio
import json

import pytest

from src.api.websocket.events import WSEventType
from src.api.websocket.manager import WebSocketManager


class _FakeWS:
    """Minimal WebSocket stub — records sent messages."""

    def __init__(self):
        self.sent: list[str] = []
        self.accepted = False

    async def accept(self):
        self.accepted = True

    async def send_text(self, text: str):
        self.sent.append(text)

    def last_event(self) -> dict | None:
        if not self.sent:
            return None
        return json.loads(self.sent[-1])


@pytest.mark.asyncio
async def test_connect_accepts_websocket():
    mgr = WebSocketManager()
    ws = _FakeWS()
    await mgr.connect(ws)
    assert ws.accepted is True
    assert mgr.connection_count == 1


@pytest.mark.asyncio
async def test_disconnect_removes_connection():
    mgr = WebSocketManager()
    ws = _FakeWS()
    await mgr.connect(ws)
    await mgr.disconnect(ws)
    assert mgr.connection_count == 0


@pytest.mark.asyncio
async def test_broadcast_sends_to_all():
    mgr = WebSocketManager()
    ws1, ws2 = _FakeWS(), _FakeWS()
    await mgr.connect(ws1)
    await mgr.connect(ws2)
    await mgr.broadcast("signal_update", {"direction": "BUY"})
    assert len(ws1.sent) == 1
    assert len(ws2.sent) == 1


@pytest.mark.asyncio
async def test_broadcast_payload_structure():
    mgr = WebSocketManager()
    ws = _FakeWS()
    await mgr.connect(ws)
    await mgr.broadcast("test_event", {"key": "val"})
    payload = json.loads(ws.sent[0])
    assert payload["event"] == "test_event"
    assert "timestamp" in payload
    assert payload["data"]["key"] == "val"


@pytest.mark.asyncio
async def test_subscription_filters_events():
    mgr = WebSocketManager()
    ws = _FakeWS()
    await mgr.connect(ws)
    # Subscribe to signal_update only
    await mgr.handle_message(ws, json.dumps({
        "type": "subscribe",
        "events": ["signal_update"],
    }))
    # Ack message goes out on subscribe
    ws.sent.clear()

    await mgr.broadcast("signal_update", {})
    await mgr.broadcast("regime_update", {})  # should be filtered

    events_received = [json.loads(m)["event"] for m in ws.sent]
    assert "signal_update" in events_received
    assert "regime_update" not in events_received


@pytest.mark.asyncio
async def test_subscription_ack_sent():
    mgr = WebSocketManager()
    ws = _FakeWS()
    await mgr.connect(ws)
    await mgr.handle_message(ws, json.dumps({
        "type": "subscribe",
        "events": ["decision_update"],
    }))
    ack = ws.last_event()
    assert ack["event"] == "subscription_ack"


@pytest.mark.asyncio
async def test_unsubscribe_restores_all_events():
    mgr = WebSocketManager()
    ws = _FakeWS()
    await mgr.connect(ws)

    # First subscribe to only one event
    await mgr.handle_message(ws, json.dumps({
        "type": "subscribe", "events": ["signal_update"]
    }))
    ws.sent.clear()

    # Unsubscribe — should receive all events again
    await mgr.handle_message(ws, json.dumps({"type": "unsubscribe"}))
    ws.sent.clear()

    await mgr.broadcast("regime_update", {})
    assert any("regime_update" == json.loads(m)["event"] for m in ws.sent)


@pytest.mark.asyncio
async def test_ping_returns_pong():
    mgr = WebSocketManager()
    ws = _FakeWS()
    await mgr.connect(ws)
    await mgr.handle_message(ws, json.dumps({"type": "ping"}))
    pong = ws.last_event()
    assert pong is not None
    assert pong["event"] == "pong"


@pytest.mark.asyncio
async def test_event_type_registry_completeness():
    assert WSEventType.SIGNAL_UPDATE  == "signal_update"
    assert WSEventType.DECISION_UPDATE == "decision_update"
    assert WSEventType.CONNECTION_ACK  == "connection_ack"
    assert len(WSEventType.ALL) >= 14


@pytest.mark.asyncio
async def test_send_to_targeted():
    mgr = WebSocketManager()
    ws1, ws2 = _FakeWS(), _FakeWS()
    await mgr.connect(ws1)
    await mgr.connect(ws2)
    await mgr.send_to(ws1, "connection_ack", {"msg": "hi"})
    assert len(ws1.sent) == 1
    assert len(ws2.sent) == 0
