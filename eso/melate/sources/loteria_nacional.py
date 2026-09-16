from __future__ import annotations

import hashlib
from io import BytesIO
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd
import requests

from ..schema import canonicalize_draws, validate_draws

# Datos Abiertos currently links the CSV from the comercializadores host.
# Keep the www host as a fallback because Loteria Nacional has used both.
OFFICIAL_HISTORY_URL = "https://comercializadores.loterianacional.gob.mx/Documentos/Historicos/Melate.csv"
OFFICIAL_HISTORY_URLS = (
    OFFICIAL_HISTORY_URL,
    "https://www.loterianacional.gob.mx/Documentos/Historicos/Melate.csv",
)
DEFAULT_UA = "Mozilla/5.0 (compatible; ESO-MelateResearch/1.0; +research)"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def parse_official_csv_bytes(raw: bytes, source_url: str = OFFICIAL_HISTORY_URL) -> pd.DataFrame:
    last_exc = None
    for enc in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            df = pd.read_csv(BytesIO(raw), encoding=enc)
            break
        except Exception as exc:  # pragma: no cover
            last_exc = exc
    else:
        raise ValueError(f"cannot parse official Melate CSV: {last_exc}")

    retrieved = datetime.now(timezone.utc).isoformat()
    df["source"] = "loteria_nacional"
    df["source_url"] = source_url
    df["source_hash"] = _sha256(raw)
    df["retrieved_at"] = retrieved
    out = canonicalize_draws(df)
    validate_draws(out, strict=True)
    return out


def load_official_file(path: str | Path, source_url: str = OFFICIAL_HISTORY_URL) -> pd.DataFrame:
    raw = Path(path).read_bytes()
    return parse_official_csv_bytes(raw, source_url=source_url)


def fetch_official_history(
    raw_dir: str | Path | None = None,
    timeout: int = 45,
    session: requests.Session | None = None,
) -> pd.DataFrame:
    s = session or requests.Session()
    errors: list[str] = []

    for url in OFFICIAL_HISTORY_URLS:
        try:
            r = s.get(url, timeout=timeout, headers={"User-Agent": DEFAULT_UA})
            r.raise_for_status()
            raw = r.content

            # Reject HTML error/challenge pages that happen to return HTTP 200.
            head = raw[:256].lstrip().lower()
            if not raw or head.startswith(b"<!doctype html") or head.startswith(b"<html"):
                raise ValueError("response is HTML, not the historical CSV")

            if raw_dir is not None:
                rd = Path(raw_dir)
                rd.mkdir(parents=True, exist_ok=True)
                stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                (rd / f"melate_official_{stamp}_{_sha256(raw)[:12]}.csv").write_bytes(raw)

            return parse_official_csv_bytes(raw, source_url=url)
        except Exception as exc:
            errors.append(f"{url}: {exc}")

    raise RuntimeError("failed to download official Melate history; " + " | ".join(errors))
