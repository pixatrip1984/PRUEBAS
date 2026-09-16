from __future__ import annotations

import hashlib
import re
import time
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timezone
from html.parser import HTMLParser
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

BASE_URL = "https://resultados.melate-e.com/melate/sorteo/{contest}"
DEFAULT_UA = "Mozilla/5.0 (compatible; ESO-MelateResearch/1.0; +research)"


class _TextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        if data and data.strip():
            self.parts.append(data.strip())


def _text(html: str) -> str:
    p = _TextExtractor()
    p.feed(html)
    return " ".join(p.parts)


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"\s+", " ", s).strip().lower()


def _money(v) -> float:
    s = str(v).replace("$", "").replace(",", "").strip()
    return float(pd.to_numeric(s, errors="coerce"))


def _int(v) -> int:
    s = re.sub(r"[^0-9-]", "", str(v))
    return int(s) if s else 0


def _parse_match(aciertos: str) -> tuple[int, bool]:
    s = _norm(aciertos)
    m = re.search(r"([2-6])\s+numero", s)
    if not m:
        raise ValueError(f"cannot parse tier match description: {aciertos!r}")
    n = int(m.group(1))
    additional = "adicional" in s
    return n, additional


def parse_prize_page(html: str, source_url: str, expected_contest: int | None = None) -> tuple[dict, pd.DataFrame]:
    body = _text(html)
    m_contest = re.search(r"sorteo numero\s*[:#]?\s*(\d+)", _norm(body))
    if not m_contest:
        m_contest = re.search(r"sorteo\s+(\d+)\s+de melate", _norm(body))
    if not m_contest:
        raise ValueError("contest id not found")
    contest = int(m_contest.group(1))
    if expected_contest is not None and contest != int(expected_contest):
        raise ValueError(f"contest mismatch: expected {expected_contest}, got {contest}")

    combo_re = re.search(
        r"combinacion ganadora .*? es:\s*([0-9,\s]+?)\.\s*con el numero adicional:\s*(\d+)",
        _norm(body),
    )
    if not combo_re:
        raise ValueError("winning combination not found")
    main = [int(x) for x in re.findall(r"\d+", combo_re.group(1))]
    if len(main) != 6:
        raise ValueError(f"expected 6 main numbers, got {main}")
    additional = int(combo_re.group(2))

    dm = re.search(r"fecha del sorteo:\s*(.*?)(?:no hubo|volante ganador|premios y numero de ganadores)", _norm(body))
    date_text = dm.group(1).strip() if dm else None

    tables = pd.read_html(StringIO(html))
    prize_table = None
    for t in tables:
        cols = [_norm(c) for c in t.columns]
        if any("aciertos" == c or "aciertos" in c for c in cols) and any("ganador" in c for c in cols):
            prize_table = t
            break

    rows = []
    if prize_table is not None:
        colmap = {_norm(c): c for c in prize_table.columns}
        c_ac = next(c for n, c in colmap.items() if "aciertos" in n)
        c_w = next(c for n, c in colmap.items() if "ganador" in n)
        c_p = next(c for n, c in colmap.items() if "premio individual" in n or "individual" in n)
        c_l = next((c for n, c in colmap.items() if "lugar" in n), None)
        for idx, r in prize_table.iterrows():
            try:
                match_natural, match_additional = _parse_match(r[c_ac])
            except ValueError:
                continue
            tier = _int(r[c_l]) if c_l is not None else idx + 1
            rows.append({
                "contest": contest,
                "tier": tier,
                "match_natural": match_natural,
                "match_additional": bool(match_additional),
                "winners": _int(r[c_w]),
                "prize_individual": _money(r[c_p]),
            })

    sha = hashlib.sha256(html.encode("utf-8", errors="ignore")).hexdigest()
    retrieved = datetime.now(timezone.utc).isoformat()
    for r in rows:
        r.update({
            "source": "melate_e",
            "source_url": source_url,
            "source_hash": sha,
            "retrieved_at": retrieved,
            "source_quality": 1,
        })
    meta = {
        "contest": contest,
        "main": tuple(main),
        "additional": additional,
        "date_text": date_text,
        "source_url": source_url,
        "source_hash": sha,
        "retrieved_at": retrieved,
    }
    return meta, pd.DataFrame(rows)


@dataclass
class MelateEClient:
    timeout: int = 30
    delay_seconds: float = 0.6
    retries: int = 3
    raw_dir: str | Path | None = None

    def __post_init__(self):
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": DEFAULT_UA})
        self.raw_dir = Path(self.raw_dir) if self.raw_dir is not None else None
        if self.raw_dir:
            self.raw_dir.mkdir(parents=True, exist_ok=True)

    def fetch_contest(self, contest: int) -> tuple[dict, pd.DataFrame]:
        url = BASE_URL.format(contest=int(contest))
        last = None
        for attempt in range(self.retries):
            try:
                r = self.session.get(url, timeout=self.timeout)
                r.raise_for_status()
                html = r.text
                if self.raw_dir:
                    (self.raw_dir / f"melate_e_{contest}.html").write_text(html, encoding="utf-8")
                return parse_prize_page(html, url, expected_contest=contest)
            except Exception as exc:
                last = exc
                if attempt + 1 < self.retries:
                    time.sleep(self.delay_seconds * (2 ** attempt))
        raise RuntimeError(f"failed contest {contest}: {last}")

    def fetch_range(self, start: int, end: int, continue_on_error: bool = True):
        metas, tiers, errors = [], [], []
        for contest in range(int(start), int(end) + 1):
            try:
                meta, df = self.fetch_contest(contest)
                metas.append(meta)
                if not df.empty:
                    tiers.append(df)
            except Exception as exc:
                errors.append({"contest": contest, "error": str(exc)})
                if not continue_on_error:
                    raise
            time.sleep(self.delay_seconds)
        tier_df = pd.concat(tiers, ignore_index=True) if tiers else pd.DataFrame()
        return pd.DataFrame(metas), tier_df, pd.DataFrame(errors)
