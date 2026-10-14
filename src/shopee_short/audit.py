"""Auditoria JSONL de rede e artefatos, sem registrar tokens ou query strings."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit
from typing import Any


class AuditLog:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path
        self.events: list[dict[str, Any]] = []
        self.request_hosts: list[str] = []
        self.shopee_hosts: list[str] = []

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat(timespec="seconds")

    def bind(self, path: Path) -> None:
        """Anexa eventos já registrados (p.ex. resolução shp.ee) ao arquivo da execução."""
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as f:
            for event in self.events:
                f.write(json.dumps(event, ensure_ascii=False) + "\n")

    def _write(self, event: dict[str, Any]) -> None:
        event = {"ts": self._now(), **event}
        self.events.append(event)
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(event, ensure_ascii=False) + "\n")

    def request(self, operation: str, url_or_host: str, method: str = "GET", status: int | None = None,
                note: str | None = None) -> None:
        if "://" in url_or_host:
            parsed = urlsplit(url_or_host)
            host = (parsed.hostname or "").lower()
            safe_path = parsed.path[:180]
        else:
            host = url_or_host.lower()
            safe_path = ""
        self.request_hosts.append(host)
        if "shopee." in host or host.endswith("susercontent.com"):
            self.shopee_hosts.append(host)
        event: dict[str, Any] = {
            "event": "network_request",
            "operation": operation,
            "method": method,
            "host": host,
            "path": safe_path,
        }
        if status is not None:
            event["status"] = status
        if note:
            event["note"] = note
        self._write(event)

    def tool_hosts(self, operation: str, hosts: list[str], note: str | None = None) -> None:
        """Registra hosts identificados em log de ferramenta de rede externa (p.ex. yt-dlp)."""
        clean = sorted({h.lower() for h in hosts if h})
        for host in clean:
            self.request_hosts.append(host)
            if "shopee." in host or host.endswith("susercontent.com"):
                self.shopee_hosts.append(host)
        self._write({"event": "external_tool_hosts", "operation": operation,
                     "hosts": clean, "note": note})

    def note(self, event: str, **fields: Any) -> None:
        self._write({"event": event, **fields})

    def summary(self) -> dict[str, Any]:
        return {
            "observed_network_hosts": sorted(set(self.request_hosts)),
            "shopee_domain_request_count": len(self.shopee_hosts),
            "shopee_domain_requests": list(self.shopee_hosts),
            "policy": "nenhuma requisição a shopee.* ou susercontent.com; só shp.ee pode ser consultado para resolver redirect",
        }
