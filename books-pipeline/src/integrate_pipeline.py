"""
merge_books_pipeline.py (Optimized & Fixed)

Pipeline para unir/normalizar/enriquecer libros desde:
 - goodreads_books.json  (newline-delimited JSON)
 - googlebooks_books.parquet OR googlebooks_books.csv

Salida:
 - landing/dim_book.parquet
 - landing/book_source_detail.parquet
 - landing/quality_metrics.json

Cambios recientes:
 - FIX: Error ArrowTypeError (mixed types float/string en IDs).
 - FIX: Manejo robusto de categorias (evita TypeError float vs list).
 - OPTIMIZACION: Lookups O(1) para ISBN y Claves Compuestas.
"""

import json
import hashlib
import os
import re
from pathlib import Path
from datetime import datetime, timezone
from difflib import SequenceMatcher
from typing import Optional, Dict, Any, List
import pandas as pd
import numpy as np

# intentar import dateutil para parseo flexible de fechas
try:
    from dateutil import parser as date_parser
except ImportError:
    date_parser = None

# --------------------------
# CONFIGURACIÓN DE CARPETAS
# --------------------------
# Raíz del proyecto (asumiendo que este script está en src/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent 

# Carpetas de datos
LANDING_DIR = PROJECT_ROOT / "landing"
STANDARD_DIR = PROJECT_ROOT / "standard"
DOCS_DIR = PROJECT_ROOT / "docs"

# Asegurar que existen
LANDING_DIR.mkdir(parents=True, exist_ok=True)
STANDARD_DIR.mkdir(parents=True, exist_ok=True)
DOCS_DIR.mkdir(parents=True, exist_ok=True)

# Inputs (Leemos desde LANDING)
GOODREADS_FILE = LANDING_DIR / "goodreads_books.json"
GOOGLE_PARQUET = LANDING_DIR / "googlebooks_books.parquet"
GOOGLE_CSV = LANDING_DIR / "googlebooks_books.csv"

# Outputs (Escribimos en STANDARD)
DIM_BOOK = STANDARD_DIR / "dim_book.parquet"
DETAIL = STANDARD_DIR / "book_source_detail.parquet"
METRICS = DOCS_DIR / "quality_metrics.json"

# --------------------------
# UTILIDADES
# --------------------------

def now_ts():
    """Devuelve timestamp UTC compatible ISO-8601."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

def safe_read_goodreads(file_path: Path) -> pd.DataFrame:
    rows = []
    if not file_path.exists():
        print(f"[WARN] No existe {file_path}")
        return pd.DataFrame(rows)

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return pd.DataFrame(rows)

def safe_read_google(file_parquet: Path, file_csv: Path) -> pd.DataFrame:
    df = pd.DataFrame()
    if file_parquet.exists():
        try:
            df = pd.read_parquet(file_parquet)
        except Exception as e:
            print(f"[WARN] Error parquet: {e}")
    elif file_csv.exists():
        try:
            df = pd.read_csv(file_csv, sep=";")
        except Exception as e:
            print(f"[WARN] Error csv: {e}")
    
    # Reemplazar NaNs por None para facilitar manejo en dicts
    if not df.empty:
        df = df.replace({np.nan: None})
    return df

def save_dataframe_robust(df: pd.DataFrame, parquet_path: Path):
    """Intenta guardar en Parquet, si falla, guarda en CSV."""
    try:
        df.to_parquet(parquet_path, index=False)
        print(f"[OK] Guardado Parquet: {parquet_path}")
    except Exception as e:
        print(f"[WARN] Falló guardado Parquet ({e}). Intentando CSV...")
        csv_path = parquet_path.with_suffix('.csv')
        # Convertir todo a string para CSV para evitar problemas
        df.to_csv(csv_path, index=False, sep=";", encoding="utf-8")
        print(f"[OK] Guardado CSV fallback: {csv_path}")

def normalize_str(val: Any) -> Optional[str]:
    if val is None: return None
    s = str(val).strip()
    # Detectar 'nan' string o float convertido
    if not s or s.lower() == 'nan': return None
    # Si termina en .0 (float convertido a string), quitarlo
    if s.endswith(".0") and s[:-2].isdigit():
        s = s[:-2]
    return re.sub(r'\s+', ' ', s)

def normalize_title(title: Optional[str]) -> Optional[str]:
    t = normalize_str(title)
    if not t: return None
    t = t.lower()
    # Eliminar puntuación, dejar letras, números y espacios
    t = re.sub(r'[^\w\s]', '', t)
    return re.sub(r'\s+', ' ', t).strip()

def normalize_author(authors_field: Any) -> List[str]:
    if not authors_field: return []
    if isinstance(authors_field, float): return [] 
    if isinstance(authors_field, str):
        return [normalize_str(a) for a in re.split(r'\||,|;', authors_field) if normalize_str(a)]
    if isinstance(authors_field, list):
        return [normalize_str(a) for a in authors_field if normalize_str(a)]
    return []

def normalize_categories(val: Any) -> List[str]:
    """Convierte input (list, str, float/nan) a lista de strings limpia."""
    if val is None: return []
    if isinstance(val, float): return [] 
    if isinstance(val, list):
        return [str(x).strip() for x in val if x and str(x).strip()]
    if isinstance(val, str):
        if "|" in val:
            return [x.strip() for x in val.split("|") if x.strip()]
        return [val.strip()]
    return []

def get_first_author(authors_field: Any) -> str:
    alist = normalize_author(authors_field)
    return alist[0] if alist else ""

def iso_date(val: Any) -> Optional[str]:
    if val is None: return None
    if isinstance(val, float): return None
    s = str(val).strip()
    if not s or s.lower() == "nan": return None
    
    if date_parser:
        try:
            dt = date_parser.parse(s, default=datetime(1,1,1))
            if dt.day != 1: return dt.date().isoformat()
            if dt.month != 1: return f"{dt.year:04d}-{dt.month:02d}"
            return f"{dt.year:04d}"
        except:
            pass
            
    m = re.match(r'^(\d{4})-(\d{1,2})-(\d{1,2})', s)
    if m: return f"{m.group(1)}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"
    m = re.match(r'^(\d{4})-(\d{1,2})', s)
    if m: return f"{m.group(1)}-{int(m.group(2)):02d}"
    m = re.match(r'^(\d{4})', s)
    if m: return m.group(1)
    return None

def normalize_currency(curr: Optional[str]) -> Optional[str]:
    if not curr: return None
    if isinstance(curr, float): return None
    curr = str(curr).strip().upper()
    mapping = {"€": "EUR", "$": "USD", "£": "GBP"}
    return mapping.get(curr, curr[:3]) if curr not in ["EUR", "USD", "GBP"] else curr

def safe_decimal(value: Any) -> Optional[float]:
    if value is None: return None
    if isinstance(value, float): return value if not np.isnan(value) else None
    try:
        return float(str(value).replace(',', '.'))
    except:
        return None

def stable_hash_id(fields: List[str]) -> str:
    concat = "||".join([f or "" for f in fields])
    return hashlib.sha1(concat.encode('utf-8')).hexdigest()

# --------------------------
# LOGICA DE MERGE
# --------------------------

def choose_survivor_value(val_good: Any, val_google: Any, prefer_source: str = "goodreads"):
    if val_good is None: return val_google
    if val_google is None: return val_good
    if str(val_good).strip() == str(val_google).strip(): return val_good
    return val_good if prefer_source == "goodreads" else val_google

def merge_records(row_good: Dict, row_google: Dict) -> Dict:
    # Normalización previa
    title_gd = normalize_str(row_good.get("title"))
    title_gg = normalize_str(row_google.get("title")) if row_google else None
    
    # Autores
    auth_gd = normalize_author(row_good.get("authors"))
    auth_gg = normalize_author(row_google.get("authors")) if row_google else []
    
    # Unificar autores
    seen = set()
    merged_authors = []
    for a in auth_gd + auth_gg:
        anorm = normalize_str(a)
        if anorm and anorm not in seen:
            seen.add(anorm)
            merged_authors.append(anorm)
    
    authors_final = " | ".join(merged_authors) if merged_authors else None
    first_author = merged_authors[0] if merged_authors else None
    
    # Categorias
    cats_gd = normalize_categories(row_good.get("categories"))
    cats_gg = normalize_categories(row_google.get("categories") if row_google else None)
    
    seen_cats = set()
    merged_cats = []
    for c in cats_gd + cats_gg:
        cnorm = normalize_str(c)
        if cnorm and cnorm not in seen_cats:
            seen_cats.add(cnorm)
            merged_cats.append(cnorm)
    categories_final = " | ".join(merged_cats) if merged_cats else None
    
    # Fechas
    pg = iso_date(row_good.get("pub_date"))
    pg2 = iso_date(row_google.get("pub_date")) if row_google else None
    pub_date = choose_survivor_value(pg, pg2)
    
    pub_year = None
    if pub_date:
        m = re.match(r'^(\d{4})', pub_date)
        if m: pub_year = int(m.group(1))

    # Precios
    p_amt = safe_decimal(row_google.get("price_amount") if row_google else None) or safe_decimal(row_good.get("price_amount"))
    p_cur = normalize_currency(row_google.get("price_currency") if row_google else None) or normalize_currency(row_good.get("price_currency"))

    # IDs (FIX: Usar normalize_str para forzar string y evitar floats)
    isbn13_g = normalize_str(row_good.get("isbn13"))
    isbn13_gg = normalize_str(row_google.get("isbn13") if row_google else None)
    isbn13 = isbn13_g or isbn13_gg

    isbn10_g = normalize_str(row_good.get("isbn"))
    isbn10_gg = normalize_str(row_google.get("isbn10") if row_google else None)
    isbn10 = isbn10_g or isbn10_gg
    
    # Title y Publisher
    title = choose_survivor_value(title_gd, title_gg)
    publisher = choose_survivor_value(normalize_str(row_good.get("publisher")), normalize_str(row_google.get("publisher") if row_google else None))

    # Mejor Calidad de los datos
    score_gd = sum(1 for v in [title_gd, auth_gd, pg, row_good.get("num_pages")] if v)
    score_gg = sum(1 for v in [title_gg, auth_gg, pg2, p_amt, isbn13_gg] if v)
    pref = "goodreads" if score_gd >= score_gg else "google"
    
    # URL preferente
    url_pref = row_good.get("url") if score_gd >= score_gg else (row_google.get("url") if row_google else row_good.get("url"))

    # Canonical ID (Ahora seguro porque isbn13 ya es str o None)
    norm_title_hash = normalize_title(title)
    cid = isbn13 if isbn13 else stable_hash_id([norm_title_hash, first_author or "", publisher or "", str(pub_year or "")])

    return {
        "canonical_id": cid,
        "isbn13": isbn13,
        "isbn10": isbn10,
        "title": title,
        "title_normalized": norm_title_hash,
        "authors": authors_final,
        "first_author": first_author,
        "publisher": publisher,
        "pub_date": pub_date,
        "pub_year": pub_year,
        "language": row_good.get("language") or (row_google.get("language") if row_google else None),
        "categories": categories_final,
        "num_pages": choose_survivor_value(row_good.get("num_pages"), row_google.get("pageCount") if row_google else None),
        "format": choose_survivor_value(row_good.get("format"), row_google.get("format") if row_google else None),
        "description": choose_survivor_value(row_good.get("desc"), row_google.get("description") if row_google else None),
        "rating_value": row_good.get("rating_value"),
        "rating_count": row_good.get("rating_count"),
        "price_amount": p_amt,
        "price_currency": p_cur,
        "source_preference": pref,
        "most_complete_url": url_pref,
        "ingestion_date_goodreads": row_good.get("ingestion_date"),
        "ingestion_date_google": row_google.get("ingestion_date") if row_google else None
    }

# --------------------------
# PIPELINE
# --------------------------

def run_pipeline():
    ts = now_ts()
    print(f"[{ts}] INICIANDO PIPELINE DE MERGE (OPTIMIZADO & FIXED)")

    # 1. Leer DataFrames
    df_good = safe_read_goodreads(GOODREADS_FILE)
    df_google = safe_read_google(GOOGLE_PARQUET, GOOGLE_CSV)
    
    print(f"[INFO] Goodreads: {len(df_good)} | Google: {len(df_google)}")

    # 2. Preparar Índices de Búsqueda (Lookups) para Google
    google_by_gbid = {}
    google_by_isbn13 = {}
    google_by_key = {} 

    if not df_google.empty:
        df_google_clean = df_google.replace({np.nan: None})
        google_records = df_google_clean.to_dict(orient="records")
        
        for rec in google_records:
            # Indexar por GB_ID
            if "gb_id" in rec and rec["gb_id"]:
                google_by_gbid[str(rec["gb_id"])] = rec
            
            # Indexar por ISBN13 (normalize para key tambien)
            isbn = normalize_str(rec.get("isbn13"))
            if isbn:
                google_by_isbn13[isbn] = rec
            
            # Indexar por Title||Author
            t_norm = normalize_title(rec.get("title"))
            a_first = get_first_author(rec.get("authors"))
            if t_norm and a_first:
                k = f"{t_norm}||{a_first}"
                if k not in google_by_key:
                    google_by_key[k] = rec

    print("[INFO] Índices de Google construidos.")

    # 3. Loop principal (Iterar Goodreads)
    merged_rows = []
    detail_rows = []
    gr_records = df_good.to_dict(orient="records") if not df_good.empty else []

    for gr in gr_records:
        matched_google = None
        
        # A) Match por ID explícito
        gid = str(gr.get("id"))
        if gid in google_by_gbid:
            matched_google = google_by_gbid[gid]
        
        # B) Match por ISBN13
        if not matched_google:
            isbn = normalize_str(gr.get("isbn13") or gr.get("isbn"))
            if isbn and isbn in google_by_isbn13:
                matched_google = google_by_isbn13[isbn]
        
        # C) Match por Clave Heurística
        if not matched_google:
            t_norm = normalize_title(gr.get("title"))
            a_first = get_first_author(gr.get("authors"))
            if t_norm and a_first:
                key = f"{t_norm}||{a_first}"
                if key in google_by_key:
                    matched_google = google_by_key[key]

        # 4. Merge
        merged = merge_records(gr, matched_google or {})
        merged_rows.append(merged)

        # 5. Detalle
        detail_rows.append({
            "canonical_id": merged["canonical_id"],
            "gb_id": gid,
            "from_google": bool(matched_google),
            "merge_method": "id" if (gid in google_by_gbid) else ("isbn" if matched_google else "heuristic"),
            "timestamp": ts
        })

    # 6. Crear DataFrame final
    df_final = pd.DataFrame(merged_rows)
    df_detail = pd.DataFrame(detail_rows)
    
    if not df_final.empty:
        # Deduplicación
        df_final["_score"] = df_final.notnull().sum(axis=1)
        df_final = df_final.sort_values("_score", ascending=False)
        df_final = df_final.drop_duplicates(subset=["canonical_id"], keep="first")
        df_final = df_final.drop(columns=["_score"])
        
        # Force string types for critical ID columns to prevent Arrow Errors
        for col in ["canonical_id", "isbn13", "isbn10"]:
             if col in df_final.columns:
                 # Convert None to empty string or just rely on astype(str) but None becomes "None"
                 # Better: fillna('') or keep None but ensure mixed types are gone.
                 # Arrow supports strings with nulls. It hates strings with floats.
                 # Our normalize_str logic should have handled this, but extra safety:
                 df_final[col] = df_final[col].astype("string")

    # 7. Guardar (Robust Save)
    save_dataframe_robust(df_final, DIM_BOOK)
    save_dataframe_robust(df_detail, DETAIL)
    
    # Metrics
    metrics = {
        "rows_input_goodreads": len(df_good),
        "rows_output": len(df_final),
        "matched_with_google": sum(1 for d in detail_rows if d["from_google"]),
        "percent_with_isbn13": round(100 * df_final["isbn13"].notnull().sum() / len(df_final), 2),
        "percent_with_isbn10": round(100 * df_final["isbn10"].notnull().sum() / len(df_final), 2),
        "percent_with_categories": round(100 * df_final["categories"].notnull().sum() / len(df_final), 2),
        "percent_with_pub_date": round(100 * df_final["pub_date"].notnull().sum() / len(df_final), 2),
        "percent_with_description": round(100 * df_final["description"].notnull().sum() / len(df_final), 2),
        "duplicates_removed": len(merged_rows) - len(df_final),  # antes de drop_duplicates
        "source_preference_counts": df_final["source_preference"].value_counts().to_dict()
    }
    
    with open(METRICS, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(f"[FIN] Proceso completado. Filas finales: {len(df_final)}")

if __name__ == "__main__":
    run_pipeline()