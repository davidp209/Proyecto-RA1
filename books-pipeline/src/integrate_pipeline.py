"""
merge_books_pipeline.py (Simplificado)

Pipeline para unir/normalizar/enriquecer libros desde:
 - landing/goodreads_books.json  (newline-delimited JSON)
 - landing/googlebooks_books.parquet OR googlebooks_books.csv

Salida:
 - standard/dim_book.parquet
 - standard/book_source_detail.parquet
 - docs/quality_metrics.json

Mantiene:
 - Idempotencia (canonical_id estable)
 - Deduplicación por calidad (_score)
 - Lookups O(1) por gb_id, isbn13 y title||author
 - Robust save Parquet + CSV
"""

import json
import hashlib
import re
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional, Any, List, Dict
import pandas as pd
import numpy as np

# Optional dateutil for flexible parsing
try:
    from dateutil import parser as date_parser
except ImportError:
    date_parser = None

# --------------------------
# CONFIG
# --------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
LANDING_DIR = PROJECT_ROOT / "landing"
STANDARD_DIR = PROJECT_ROOT / "standard"
DOCS_DIR = PROJECT_ROOT / "docs"

LANDING_DIR.mkdir(parents=True, exist_ok=True)
STANDARD_DIR.mkdir(parents=True, exist_ok=True)
DOCS_DIR.mkdir(parents=True, exist_ok=True)

GOODREADS_FILE = LANDING_DIR / "goodreads_books.json"
GOOGLE_PARQUET = LANDING_DIR / "googlebooks_books.parquet"
GOOGLE_CSV = LANDING_DIR / "googlebooks_books.csv"

DIM_BOOK = STANDARD_DIR / "dim_book.parquet"
DETAIL = STANDARD_DIR / "book_source_detail.parquet"
METRICS = DOCS_DIR / "quality_metrics.json"

# --------------------------
# UTIL
# --------------------------

def now_ts() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

def safe_read_goodreads(path: Path) -> pd.DataFrame:
    if not path.exists():
        print(f"[WARN] No existe {path}")
        return pd.DataFrame()
    rows = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                # ignorar lineas corruptas
                continue
    return pd.DataFrame(rows)

def safe_read_google(parquet: Path, csv: Path) -> pd.DataFrame:
    df = pd.DataFrame()
    if parquet.exists():
        try:
            df = pd.read_parquet(parquet)
        except Exception as e:
            print(f"[WARN] Error leyendo parquet {parquet}: {e}")
    elif csv.exists():
        try:
            df = pd.read_csv(csv, sep=";")
        except Exception as e:
            print(f"[WARN] Error leyendo csv {csv}: {e}")
    if not df.empty:
        df = df.replace({np.nan: None})
    return df

def save_dataframe_robust(df: pd.DataFrame, path: Path):
    """Guarda Parquet (si puede) y siempre escribe CSV como respaldo."""
    if df is None or df.empty:
        print(f"[WARN] DF vacío, no guardado: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    csv_path = path.with_suffix(".csv")
    try:
        df.to_parquet(path, index=False)
        print(f"[OK] Parquet: {path}")
    except Exception as e:
        print(f"[WARN] No se pudo escribir Parquet ({e}).")
    try:
        df.to_csv(csv_path, index=False, sep=";", encoding="utf-8")
        print(f"[OK] CSV: {csv_path}")
    except Exception as e:
        print(f"[ERROR] No se pudo escribir CSV ({e}).")

# --------------------------
# NORMALIZACIÓN (compacta)
# --------------------------

def normalize_str(val: Any) -> Optional[str]:
    if val is None:
        return None
    s = str(val).strip()
    if not s or s.lower() == "nan":
        return None
    # quitar ".0" en floats serializados como strings ("123.0")
    if s.endswith(".0") and s[:-2].isdigit():
        s = s[:-2]
    return re.sub(r"\s+", " ", s)

def normalize_title(title: Any) -> Optional[str]:
    t = normalize_str(title)
    if not t:
        return None
    t = t.lower()
    t = re.sub(r"[^\w\s]", "", t)
    return re.sub(r"\s+", " ", t).strip()

def normalize_author(auth: Any) -> List[str]:
    if not auth or isinstance(auth, float):
        return []
    if isinstance(auth, str):
        parts = re.split(r"[|,;]", auth)
    elif isinstance(auth, list):
        parts = auth
    else:
        return []
    return [normalize_str(p) for p in parts if normalize_str(p)]

def normalize_categories(val: Any) -> List[str]:
    if not val or isinstance(val, float):
        return []
    if isinstance(val, list):
        return [str(x).strip() for x in val if x and str(x).strip()]
    s = str(val)
    if "|" in s:
        return [x.strip() for x in s.split("|") if x.strip()]
    return [s.strip()]

def get_first_author(auth: Any) -> str:
    a = normalize_author(auth)
    return a[0] if a else ""

def iso_date(val: Any) -> Optional[str]:
    if not val or isinstance(val, float):
        return None
    s = str(val).strip()
    if not s or s.lower() == "nan":
        return None
    # try dateutil
    if date_parser:
        try:
            dt = date_parser.parse(s, default=datetime(1,1,1))
            if dt.day != 1:
                return dt.date().isoformat()
            if dt.month != 1:
                return f"{dt.year:04d}-{dt.month:02d}"
            return f"{dt.year:04d}"
        except Exception:
            pass
    # fallback patterns
    m = re.match(r"^(\d{4})-(\d{1,2})-(\d{1,2})", s)
    if m:
        return f"{m.group(1)}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"
    m = re.match(r"^(\d{4})-(\d{1,2})", s)
    if m:
        return f"{m.group(1)}-{int(m.group(2)):02d}"
    m = re.match(r"^(\d{4})", s)
    if m:
        return m.group(1)
    return None

def normalize_currency(curr: Any) -> Optional[str]:
    if not curr or isinstance(curr, float):
        return None
    s = str(curr).strip().upper()
    mapping = {"€": "EUR", "$": "USD", "£": "GBP"}
    return mapping.get(s, s[:3])

def safe_decimal(v: Any) -> Optional[float]:
    if v is None:
        return None
    if isinstance(v, float):
        return v if not np.isnan(v) else None
    try:
        return float(str(v).replace(",", "."))
    except Exception:
        return None

def stable_hash_id(fields: List[str]) -> str:
    return hashlib.sha1("||".join(f or "" for f in fields).encode("utf-8")).hexdigest()

# --------------------------
# MERGE LOGIC
# --------------------------

def choose_survivor(val_g: Any, val_gg: Any, prefer: str = "goodreads"):
    if val_g is None:
        return val_gg
    if val_gg is None:
        return val_g
    if str(val_g).strip() == str(val_gg).strip():
        return val_g
    return val_g if prefer == "goodreads" else val_gg

def merge_records(g: Dict, gg: Dict) -> Dict:
    # normalize basics
    t_g = normalize_str(g.get("title"))
    t_gg = normalize_str(gg.get("title")) if gg else None

    # authors merge
    a_g = normalize_author(g.get("authors"))
    a_gg = normalize_author(gg.get("authors")) if gg else []
    seen = set()
    merged_authors = []
    for a in a_g + a_gg:
        n = normalize_str(a)
        if n and n not in seen:
            seen.add(n); merged_authors.append(n)
    authors_str = " | ".join(merged_authors) if merged_authors else None
    first_author = merged_authors[0] if merged_authors else None

    # categories
    cats_g = normalize_categories(g.get("categories"))
    cats_gg = normalize_categories(gg.get("categories") if gg else None)
    seen_cats = set()
    merged_cats = []
    for c in cats_g + cats_gg:
        cn = normalize_str(c)
        if cn and cn not in seen_cats:
            seen_cats.add(cn); merged_cats.append(cn)
    categories_str = " | ".join(merged_cats) if merged_cats else None

    # dates
    pub_g = iso_date(g.get("pub_date"))
    pub_gg = iso_date(gg.get("pub_date")) if gg else None
    pub_date = choose_survivor(pub_g, pub_gg)

    pub_year = None
    if pub_date:
        m = re.match(r"^(\d{4})", pub_date)
        if m:
            pub_year = int(m.group(1))

    # prices
    price_amt = safe_decimal(gg.get("price_amount") if gg else None) or safe_decimal(g.get("price_amount"))
    price_cur = normalize_currency(gg.get("price_currency") if gg else None) or normalize_currency(g.get("price_currency"))

    # ids
    isbn13 = normalize_str(g.get("isbn13")) or normalize_str(gg.get("isbn13") if gg else None)
    # isbn10 from both possible keys 'isbn10' or legacy 'isbn'
    isbn10 = normalize_str(g.get("isbn10") or g.get("isbn")) or normalize_str(gg.get("isbn10") if gg else None)

    title = choose_survivor(t_g, t_gg)
    publisher = choose_survivor(normalize_str(g.get("publisher")), normalize_str(gg.get("publisher") if gg else None))

    # score heuristic to pick preferred source
    score_g = sum(1 for v in [t_g, a_g, pub_g, g.get("num_pages")] if v)
    score_gg = sum(1 for v in [t_gg, a_gg, pub_gg, price_amt, normalize_str(gg.get("isbn13") if gg else None)] if v)
    pref = "goodreads" if score_g >= score_gg else "google"
    url_pref = g.get("url") if score_g >= score_gg else (gg.get("url") if gg else g.get("url"))

    norm_title_hash = normalize_title(title)
    cid = isbn13 if isbn13 else stable_hash_id([norm_title_hash or "", first_author or "", publisher or "", str(pub_year or "")])

    return {
        "canonical_id": cid,
        "isbn13": isbn13,
        "isbn10": isbn10,
        "title": title,
        "title_normalized": norm_title_hash,
        "authors": authors_str,
        "first_author": first_author,
        "publisher": publisher,
        "pub_date": pub_date,
        "pub_year": pub_year,
        "language": g.get("language") or (gg.get("language") if gg else None),
        "categories": categories_str,
        "num_pages": choose_survivor(g.get("num_pages"), gg.get("pageCount") if gg else None),
        "format": choose_survivor(g.get("format"), gg.get("format") if gg else None),
        "description": choose_survivor(g.get("desc"), gg.get("description") if gg else None),
        "rating_value": g.get("rating_value"),
        "rating_count": g.get("rating_count"),
        "price_amount": price_amt,
        "price_currency": price_cur,
        "source_preference": pref,
        "most_complete_url": url_pref,
        "ingestion_date_goodreads": g.get("ingestion_date"),
        "ingestion_date_google": gg.get("ingestion_date") if gg else None
    }

# --------------------------
# PIPELINE
# --------------------------

def run_pipeline():
    ts = now_ts()
    print(f"[{ts}] INICIANDO PIPELINE DE MERGE (SIMPLIFICADO)")

    df_good = safe_read_goodreads(GOODREADS_FILE)
    df_gg = safe_read_google(GOOGLE_PARQUET, GOOGLE_CSV)

    print(f"[INFO] Goodreads: {len(df_good)} | Google: {len(df_gg)}")

    # Build lookups for google records (O(1) lookups)
    google_by_gbid = {}
    google_by_isbn13 = {}
    google_by_key = {}

    if not df_gg.empty:
        recs = df_gg.replace({np.nan: None}).to_dict(orient="records")
        for r in recs:
            if r.get("gb_id"):
                google_by_gbid[str(r["gb_id"])] = r
            isbn = normalize_str(r.get("isbn13"))
            if isbn:
                google_by_isbn13[isbn] = r
            tnorm = normalize_title(r.get("title"))
            afirst = get_first_author(r.get("authors"))
            if tnorm and afirst:
                k = f"{tnorm}||{afirst}"
                google_by_key.setdefault(k, r)

    print("[INFO] Índices Google construidos.")

    merged_rows = []
    detail_rows = []
    gr_recs = df_good.replace({np.nan: None}).to_dict(orient="records") if not df_good.empty else []

    for g in gr_recs:
        matched = None
        gid = str(g.get("id")) if g.get("id") is not None else ""
        if gid and gid in google_by_gbid:
            matched = google_by_gbid[gid]
            method = "id"
        else:
            isbn = normalize_str(g.get("isbn13") or g.get("isbn"))
            if isbn and isbn in google_by_isbn13:
                matched = google_by_isbn13[isbn]
                method = "isbn"
            else:
                tnorm = normalize_title(g.get("title"))
                afirst = get_first_author(g.get("authors"))
                key = f"{tnorm}||{afirst}" if tnorm and afirst else None
                if key and key in google_by_key:
                    matched = google_by_key[key]
                    method = "heuristic"
                else:
                    method = "none"

        merged = merge_records(g, matched or {})
        merged_rows.append(merged)

        detail_rows.append({
            "canonical_id": merged["canonical_id"],
            "gb_id": gid,
            "from_google": bool(matched),
            "merge_method": method,
            "timestamp": ts
        })

    # Final DF
    df_final = pd.DataFrame(merged_rows)
    df_detail = pd.DataFrame(detail_rows)

    if not df_final.empty:
        # Deduplicate by canonical_id preferring more-complete rows
        df_final["_score"] = df_final.notnull().sum(axis=1)
        df_final = df_final.sort_values("_score", ascending=False)
        df_final = df_final.drop_duplicates(subset=["canonical_id"], keep="first").drop(columns=["_score"])

        # Force string types for id columns (avoid mixed types / pyarrow errors)
        for c in ["canonical_id", "isbn13", "isbn10"]:
            if c in df_final.columns:
                df_final[c] = df_final[c].astype("string")

    # Save outputs
    save_dataframe_robust(df_final, DIM_BOOK)
    save_dataframe_robust(df_detail, DETAIL)

    # Metrics
    total_in = len(df_good)
    total_out = len(df_final) if not df_final.empty else 0
    matched_with_google = sum(1 for d in detail_rows if d.get("from_google"))

    metrics = {
        "generated_at": ts,
        "rows_input_goodreads": total_in,
        "rows_output": total_out,
        "matched_with_google": matched_with_google,
        "percent_with_isbn13": round(100 * df_final["isbn13"].notnull().sum() / total_out, 2) if total_out else 0,
        "percent_with_isbn10": round(100 * df_final["isbn10"].notnull().sum() / total_out, 2) if total_out else 0,
        "percent_with_categories": round(100 * df_final["categories"].notnull().sum() / total_out, 2) if total_out else 0,
        "percent_with_pub_date": round(100 * df_final["pub_date"].notnull().sum() / total_out, 2) if total_out else 0,
        "duplicates_removed": len(merged_rows) - total_out,
        "source_preference_counts": df_final["source_preference"].value_counts().to_dict() if total_out else {}
    }

    with open(METRICS, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"[OK] Metrics saved: {METRICS}")

    print(f"[FIN] Proceso completado. Filas finales: {total_out}")

if __name__ == "__main__":
    run_pipeline()
