import json
import time
import requests
import pandas as pd
from difflib import SequenceMatcher
from pathlib import Path

# ============================================
# 1) CONFIGURACIÓN
# ============================================

BASE_DIR = Path(__file__).resolve().parent.parent / "landing"
INPUT_FILE = BASE_DIR / "goodreads_books.json"
OUTPUT_FILE = BASE_DIR / "googlebooks_books.csv"

# Sesión HTTP persistente (más rápida y eficiente)
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "Mozilla/5.0"})

# Endpoint de Google Books
API_URL = "https://www.googleapis.com/books/v1/volumes"


# ============================================
# 2) UTILIDADES DE SIMILITUD
# ============================================
 #   Calcula una similitud entre 0 y 1 usando SequenceMatcher.
 #   Se usa para comparar títulos y autores.
    

def similarity(a, b):
    if not a or not b:
        return 0
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


# ============================================
# 3) DESCARGAR TODOS LOS RESULTADOS DE GOOGLE BOOKS
# ============================================

def search_api_all(query):
    """
    Recupera TODOS los resultados posibles de Google Books.
    Usa paginación (startIndex) hasta que no haya más.
    
    Google Books permite máximo 40 resultados por petición.
    """
    results = []
    start = 0
    max_results = 40  # límite del API

    while True:
        try:
            # Petición
            res = SESSION.get(
                API_URL,
                params={
                    "q": query,
                    "printType": "books",
                    "maxResults": max_results,
                    "startIndex": start
                },
                timeout=5
            )

            # Rate limit → espera y reintenta
            if res.status_code == 429:
                time.sleep(2)
                continue

            # Error → salimos
            if res.status_code != 200:
                break

            data = res.json()
            items = data.get("items", [])

            # Si no hay más resultados → terminamos
            if not items:
                break

            results.extend(items)

            # Si recibimos menos de 40, no hay más páginas
            if len(items) < max_results:
                break

            # Siguiente página
            start += max_results
            time.sleep(0.3)

        except Exception:
            break

    return results


# ============================================
# 4) ELEGIR EL MEJOR RESULTADO (MATCH)
# ============================================

def choose_best_result(gr_book, results):
    """
    Evalúa todos los resultados devueltos por Google Books
    y selecciona el que más se parece al libro de Goodreads.

    Criterios usados (modelo recomendado):
      +100 si ISBN_13 coincide EXACTO
      +80  si ISBN_10 coincide EXACTO
      +50 * similitud de título
      +30 * similitud de autor
    """
    best = None
    best_score = 0

    # Campos del libro de Goodreads
    gr_title = gr_book.get("title", "")
    gr_author = (gr_book.get("authors") or [""])[0]
    gr_isbn13 = gr_book.get("isbn13")
    gr_isbn10 = None
    if gr_book.get("isbn13") and len(gr_book["isbn13"]) == 10:
        gr_isbn10 = gr_book["isbn13"]

    # Evaluar cada resultado de Google
    for item in results:
        vol = item.get("volumeInfo", {})
        ids = {
            x.get("type"): x.get("identifier")
            for x in vol.get("industryIdentifiers", [])
        }

        score = 0

        # 1) Coincidencia EXACTA de ISBN (criterio más fuerte)
        if gr_isbn13 and ids.get("ISBN_13") == gr_isbn13:
            score += 100

        if gr_isbn10 and ids.get("ISBN_10") == gr_isbn10:
            score += 80

        # 2) Similaridad de título
        score += similarity(gr_title, vol.get("title", "")) * 50

        # 3) Similaridad de autor
        authors_list = vol.get("authors", [""])
        if authors_list:
            score += similarity(gr_author, authors_list[0]) * 30

        # Guardar el mejor resultado encontrado
        if score > best_score:
            best_score = score
            best = item

    return best


# ============================================
# 5) EXTRAER DATOS LIMPIOS DEL RESULTADO GOOGLE BOOKS
# ============================================

def extract_data(gr_id, item):
    """
    Convierte un resultado de Google Books a un diccionario limpio.
    Si no encuentra nada, marca NOT_FOUND.
    """
    if not item:
        return {"gb_id": gr_id, "google_id": "NOT_FOUND"}

    vol = item.get("volumeInfo", {})
    sale = item.get("saleInfo", {})

    ids = {x.get("type"): x.get("identifier")
           for x in vol.get("industryIdentifiers", [])}

    price = sale.get("listPrice") or sale.get("retailPrice") or {}

    return {
        "gb_id": gr_id,
        "google_id": item.get("id"),
        "title": vol.get("title"),
        "authors": " | ".join(vol.get("authors", [])),
        "publisher": vol.get("publisher"),
        "pub_date": vol.get("publishedDate"),
        "categories": " | ".join(vol.get("categories", [])) if vol.get("categories") else None,
        "isbn13": ids.get("ISBN_13"),
        "price_amount": price.get("amount"),
        "price_currency": price.get("currencyCode")
    }


# ============================================
# 6) PROCESO PRINCIPAL (CORREGIDO Y FINAL)
# ============================================

def main():
    if not INPUT_FILE.exists():
        print("ERROR: no existe goodreads_books.json")
        return

    # ----------------------------------------
    # A) Cargar IDs ya procesados (idempotencia)
    # ----------------------------------------
    processed_ids = set()
    if OUTPUT_FILE.exists():
        try:
            df_old = pd.read_csv(OUTPUT_FILE, sep=";", dtype=str)
            processed_ids = set(df_old["gb_id"])
        except:
            pass

    # ----------------------------------------
    # B) Cargar todos los libros de Goodreads
    # ----------------------------------------
    books = []
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            b = json.loads(line)
            if b["id"] not in processed_ids:
                books.append(b)

    print(f"Se procesarán {len(books)} libros nuevos de Goodreads")

    if not books:
        print("Todo al día, no hay libros nuevos.")
        return

    # ----------------------------------------
    # C) Procesar cada libro de Goodreads
    # ----------------------------------------
    results = []

    for i, book in enumerate(books, 1):

        title = book.get("title")
        first_author = (book.get("authors") or [""])[0]
        isbn13 = book.get("isbn13")

        # Estrategias de búsqueda
        queries = [
            f"isbn:{isbn13}" if isbn13 else None,
            f'intitle:"{title}" inauthor:"{first_author}"',
            f'intitle:"{title}"'
        ]

        google_item = None

        # --- Bucle de Queries Corregido ---
        for q in queries:
            if not q:
                continue

            all_results = search_api_all(q)

            if all_results:
                # 1. Buscamos si hay un candidato bueno en esta lista
                candidate = choose_best_result(book, all_results)
                
                # 2. SOLO paramos si el candidato es válido.
                # Si candidate es None (porque los resultados eran malos), seguimos probando queries.
                if candidate:
                    google_item = candidate
                    break 
        # ----------------------------------

        print(f"[{i}/{len(books)}] {title} → "
              f"{'Encontrado' if google_item else 'NO encontrado'}")

        results.append(extract_data(book["id"], google_item))
        time.sleep(0.5) # Pausa para respetar a la API

    # ----------------------------------------
    # D) Guardar resultados en CSV
    # ----------------------------------------
    if results:
        df = pd.DataFrame(results)
        cols = [
            "gb_id", "google_id", "title", "authors", "publisher",
            "pub_date", "categories", "isbn13",
            "price_amount", "price_currency"
        ]

        df.to_csv(
            OUTPUT_FILE,
            sep=";",
            mode="a",
            index=False,
            header=not OUTPUT_FILE.exists(),
            columns=cols
        )

        print("Guardado completado.")

# ============================================
# EJECUCIÓN
# ============================================

if __name__ == "__main__":
    main()
