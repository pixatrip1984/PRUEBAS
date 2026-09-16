from __future__ import annotations

import hashlib
from io import BytesIO
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd
import requests

from ..schema import canonicalize_draws, validate_draws

OFFICIAL_HISTORY_URL = "https://www.loterianacional.gob.mx/Documentos/Historicos/Melate.csv"
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
    r = s.get(OFFICIAL_HISTORY_URL, timeout=timeout, headers={"User-Agent": DEFAULT_UA})
    r.raise_for_status()
    raw = r.content
    if raw_dir is not None:
        raw_dir = Path(raw_dir)
        raw_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        (raw_dir / f"melate_official_{stamp}_{_sha256(raw)[:12]}.csv").write_bytes(raw)
    return parse_official_csv_bytes(raw)
