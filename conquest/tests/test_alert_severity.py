"""Severity gating for conquest.production.alerts.attach_alert.

The v5 alert refactor changed _alert(...) so that severity='info' (the new
default) writes only a [ALERT] log line; severity='critical' additionally
fires email. The bulk of existing call sites take the default and stop
emailing; only the four DD-related alerts pass severity='critical'.
"""
from __future__ import annotations

import pytest

from conquest.production.alerts import attach_alert


class _StubNotify:
    def __init__(self) -> None:
        self.email_calls: list[tuple[str, str, str]] = []

    def email(self, to: str, subject: str, body: str) -> None:
        self.email_calls.append((to, subject, body))


class _StubAlgo:
    """Duck-typed stand-in for a Lean QCAlgorithm. attach_alert reads only
    .get_parameter(), .log(), and .notify on the passed object."""

    def __init__(self, alert_email: str = "test@example.com") -> None:
        self._params = {"ALERT_EMAIL": alert_email}
        self.log_lines: list[str] = []
        self.notify = _StubNotify()

    def get_parameter(self, key: str) -> str:
        return self._params.get(key, "")

    def log(self, line: str) -> None:
        self.log_lines.append(line)


@pytest.fixture
def algo():
    a = _StubAlgo()
    attach_alert(a)
    return a


def test_default_severity_is_log_only(algo: _StubAlgo) -> None:
    algo._alert("REGIME TRANSITION", "Deflation -> Inflation")
    assert any("[ALERT] REGIME TRANSITION" in line for line in algo.log_lines)
    assert algo.notify.email_calls == []


def test_severity_info_is_log_only(algo: _StubAlgo) -> None:
    algo._alert("VIX CASH GATE ENTERED", "VIX=31.2", severity="info")
    assert any("[ALERT] VIX CASH GATE ENTERED" in line for line in algo.log_lines)
    assert algo.notify.email_calls == []


def test_severity_critical_logs_and_emails(algo: _StubAlgo) -> None:
    algo._alert("PORTFOLIO DD > 20%", "dd=-21.3%", severity="critical")
    assert any("[ALERT] PORTFOLIO DD > 20%" in line for line in algo.log_lines)
    assert len(algo.notify.email_calls) == 1
    to, subject, body = algo.notify.email_calls[0]
    assert to == "test@example.com"
    assert subject == "[conquest] PORTFOLIO DD > 20%"
    assert body == "dd=-21.3%"


def test_critical_alert_helper_emails(algo: _StubAlgo) -> None:
    algo._critical_alert("PORTFOLIO DD CIRCUIT ON", "dd=-50%")
    assert len(algo.notify.email_calls) == 1
    assert "[ALERT] PORTFOLIO DD CIRCUIT ON" in algo.log_lines[-1]


def test_dedup_key_suppresses_repeat(algo: _StubAlgo) -> None:
    algo._alert("DATA STALE", "regime=132d", dedup_key="data_stale")
    algo._alert("DATA STALE", "regime=132d", dedup_key="data_stale")
    alert_lines = [L for L in algo.log_lines if "[ALERT]" in L]
    assert len(alert_lines) == 1


def test_dedup_key_lets_different_body_through(algo: _StubAlgo) -> None:
    algo._alert("DATA STALE", "regime=132d", dedup_key="data_stale")
    algo._alert("DATA STALE", "regime=200d", dedup_key="data_stale")
    alert_lines = [L for L in algo.log_lines if "[ALERT]" in L]
    assert len(alert_lines) == 2


def test_no_alert_email_means_no_email_even_critical() -> None:
    a = _StubAlgo(alert_email="")
    attach_alert(a)
    a._alert("PORTFOLIO DD > 20%", "dd=-21%", severity="critical")
    assert a.notify.email_calls == []
    assert any("[ALERT] PORTFOLIO DD > 20%" in L for L in a.log_lines)


def test_attach_alert_is_idempotent() -> None:
    """If _alert is already present on the algo (project-specific override),
    attach_alert should skip without overwriting."""
    a = _StubAlgo()

    def sentinel(*args, **kwargs):
        a.log_lines.append("sentinel")
        return None

    a._alert = sentinel
    attach_alert(a)
    assert a._alert is sentinel
