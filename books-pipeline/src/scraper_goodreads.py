import requests
import json
import time
import random
import re
from bs4 import BeautifulSoup
from dataclasses import dataclass, field, asdict
from typing import List, Optional
from datetime import datetime
from pathlib import Path

# --- 1. CONFIGURACIÓN ---
CURRENT_DIR = Path(__file__).resolve().parent 
PROJECT_ROOT = CURRENT_DIR.parent
LANDING_DIR = PROJECT_ROOT / "landing"
LANDING_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE = LANDING_DIR / "goodreads_books.json"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9"
}
BASE_BOOK_URL = "https://www.goodreads.com/book/show/"
SEARCH_URL = "https://www.goodreads.com/search"

# --- 2. DATACLASS ---
@dataclass
class BookData:
    id: str
    isbn10: Optional[str] = None       
    isbn13: Optional[str] = None
    title: Optional[str] = None
    authors: List[str] = field(default_factory=list)
    publisher: Optional[str] = None
    pub_date: Optional[str] = None
    language: Optional[str] = None
    categories: List[str] = field(default_factory=list)
    desc: Optional[str] = None
    num_pages: Optional[int] = None
    format: Optional[str] = None
    rating_value: Optional[float] = None
    rating_count: Optional[int] = None
    url: str = ""
    ingestion_date: Optional[str] = None


# --- 3. FUNCIONES DE LIMPIEZA ---
def clean_text_deep(text):
    """Limpia descripciones HTML complejas"""
    if not text: return None
    text = text.replace("<br>", "\n").replace("<br />", "\n")
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text(separator="\n").strip()

def extract_pages_from_html(soup):
    """Intenta extraer el número de páginas del HTML si no está en el JSON"""
    try:
        p_tag = soup.find("p", {"data-testid": "pagesFormat"})
        if p_tag:
            match = re.search(r'(\d+)', p_tag.get_text())
            if match: return int(match.group(1))
    except: pass
    return None

def extract_publisher_info(soup, json_publisher):
    """Estrategia híbrida para sacar el Publisher"""
    if json_publisher: return json_publisher
    try:
        pub_tag = soup.find("p", {"data-testid": "publicationInfo"})
        if pub_tag:
            text = pub_tag.get_text()
            if " by " in text:
                return text.split(" by ")[-1].strip()
    except: pass
    return None

# --- 4. SCRAPER PRINCIPAL ---
def get_book_details(book_id):
    url = BASE_BOOK_URL + str(book_id)
    try:
        resp = requests.get(url, headers=HEADERS)
        if resp.status_code != 200:
            return None
    except Exception:
        return None

    html = resp.text
    soup = BeautifulSoup(html, "html.parser")

    # --- A. JSON-LD (Datos estructurados ocultos) ---
    book_json = {}
    script_tag = soup.find("script", {"type": "application/ld+json"})
    if script_tag:
        try:
            data = json.loads(script_tag.string)
            if isinstance(data, list):
                for item in data:
                    if item.get("@type") == "Book":
                        book_json = item
                        break
            elif data.get("@type") == "Book":
                book_json = data
        except:
            pass

    # Crear objeto BookData inicial
    bd = BookData(
        id=str(book_id),
        title=book_json.get('name') or (soup.find("meta", property="og:title") or {}).get("content"),
        desc=clean_text_deep(book_json.get('description') or
                             (soup.find("div", {"data-testid": "description"}) or {}).get_text(separator="\n")),
        authors=[],
        categories=[],
        url=url,
        ingestion_date=datetime.now().isoformat()
    )

    # Autores
    raw_author = book_json.get('author')
    if raw_author:
        if isinstance(raw_author, list):
            bd.authors = [a.get('name') for a in raw_author if isinstance(a, dict)]
        elif isinstance(raw_author, dict):
            if raw_author.get('name'):
                bd.authors = [raw_author.get('name')]

    # Categorías
    cats_list = []
    for link in soup.find_all("a", href=re.compile(r'/genres/')):
        g = link.get_text(strip=True)
        if g and len(g) > 2 and g not in cats_list:
            cats_list.append(g)
    bd.categories = list(set(cats_list))[:5]

    # Formato y páginas
    bd.num_pages = book_json.get('numberOfPages') or extract_pages_from_html(soup)
    bd.format = book_json.get('bookFormat')

    # Ratings
    if book_json.get('aggregateRating'):
        bd.rating_value = float(book_json.get('aggregateRating', {}).get('ratingValue', 0))
        bd.rating_count = int(book_json.get('aggregateRating', {}).get('ratingCount', 0))

    # Fecha de publicación inicial y publisher
    bd.pub_date = book_json.get('datePublished')
    json_pub_name = book_json.get('publisher', {}).get('name') if isinstance(book_json.get('publisher'), dict) else None
    bd.publisher = extract_publisher_info(soup, json_pub_name)

    # --- B. JSON embebido (opcional, para ISBN-10 y publisher alternativo) ---
    try:
        match = re.search(r'"details"\s*:\s*({.*?})\s*,\s*"', html, flags=re.DOTALL)
        if match:
            details = json.loads(match.group(1))
            # Publisher
            publisher = details.get("publisher")
            if isinstance(publisher, dict):
                bd.publisher = publisher.get("name") or bd.publisher
            elif isinstance(publisher, str):
                bd.publisher = publisher.strip() or bd.publisher

            # ISBNs
            isbn10 = details.get("isbn")
            isbn13 = details.get("isbn13")
            if isbn10 and len(isbn10) == 10:
                bd.isbn10 = isbn10
            if isbn13 and len(isbn13) == 13:
                bd.isbn13 = isbn13

            # Formato y páginas alternativos
            if not bd.format:
                bd.format = details.get("format")
            if not bd.num_pages:
                bd.num_pages = details.get("numPages")

            # Fecha de publicación desde timestamp
            ts = details.get("publicationTime")
            if ts:
                bd.pub_date = datetime.fromtimestamp(ts / 1000).strftime("%Y-%m-%d")

            # Idioma
            lang = details.get("language")
            if isinstance(lang, dict):
                bd.language = lang.get("name", "").lower().strip()
            elif isinstance(lang, str):
                bd.language = lang.lower().strip()
    except:
        pass

    # --- C. Último intento de ISBN desde la página si sigue faltando ---
    if not bd.isbn10 or not bd.isbn13:
        try:
            info_rows = soup.find_all("div", class_="infoBoxRowItem")
            for div in info_rows:
                text = div.get_text(strip=True)
                if re.match(r'^\d{10}$', text) and not bd.isbn10:
                    bd.isbn10 = text
                elif re.match(r'^\d{13}$', text) and not bd.isbn13:
                    bd.isbn13 = text
        except:
            pass

    return bd


def get_book_ids_from_search(query, target_count=15):
    """
    Busca páginas sucesivamente (page=1, page=2...) hasta encontrar 'target_count' libros únicos.
    """
    found_ids = []
    current_page = 1
    
    print(f"--- Buscando '{query}' (Objetivo: {target_count} libros) ---")
    
    while len(found_ids) < target_count:
        print(f" -> Escrapeando página de búsqueda {current_page}...")
        
        params = {'q': query, 'page': current_page, 'search_type': 'books'}
        
        try:
            resp = requests.get(SEARCH_URL, headers=HEADERS, params=params)
            if resp.status_code != 200:
                print("Error en la búsqueda o bloqueo de IP.")
                break
                
            soup = BeautifulSoup(resp.text, "html.parser")
            links = soup.find_all("a", class_="bookTitle")
            
            if not links:
                print(" -> No hay más resultados.")
                break
            
            new_ids_found_on_page = 0
            for link in links:
                match = re.search(r'/show/(\d+)', link.get('href'))
                if match:
                    bid = match.group(1)
                    if bid not in found_ids:
                        found_ids.append(bid)
                        new_ids_found_on_page += 1
                        
                        if len(found_ids) >= target_count:
                            break
            
            print(f"    Encontrados {new_ids_found_on_page} nuevos en esta página.")
            
            current_page += 1
            time.sleep(1) # Pausa breve

        except Exception as e:
            print(f"Error: {e}")
            break
            
    return found_ids[:target_count]

# --- 5. EJECUCIÓN ROBUSTA ---
if __name__ == "__main__":
    TERMINO = "Data Science"
    CANTIDAD_OBJETIVO = 15 
    
    # A) Asegurar que el archivo existe (para tus pruebas de borrado)
    if not OUTPUT_FILE.exists():
        OUTPUT_FILE.touch() 
        print(f"> Archivo '{OUTPUT_FILE.name}' no existía. Creado vacío.")
    
    # B) Cargar IDs previos (Deduplicación)
    existing_ids = set()
    with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    existing_ids.add(json.loads(line)['id'])
                except: pass
    
    print(f"> IDs ya existentes en el archivo: {len(existing_ids)}")
    
    # C) Buscar IDs nuevos
    ids = get_book_ids_from_search(TERMINO, target_count=CANTIDAD_OBJETIVO)
    ids_to_scrape = [bid for bid in ids if bid not in existing_ids]
    
    print(f"\n>>> Se procesarán {len(ids_to_scrape)} libros nuevos.\n")

    # D) Guardar en modo Append
    if ids_to_scrape:
        with open(OUTPUT_FILE, 'a', encoding='utf-8') as f:
            for i, bid in enumerate(ids_to_scrape):
                # Doble check
                if bid in existing_ids: continue

                bk = get_book_details(bid)
                
                if bk and bk.title:
                    f.write(json.dumps(asdict(bk), ensure_ascii=False) + "\n")
                    existing_ids.add(bid)
                    print(f"[{i+1}/{len(ids_to_scrape)}] Guardado: {bk.title[:30]}...")
                else:
                    print(f"[{i+1}] Error recuperando detalles.")
                
                time.sleep(random.uniform(1.0, 2.0))
    else:
        print(">>> ¡Objetivo cumplido! Ya tenías todos estos libros.")

    print(f"\n>>> FINALIZADO. Total libros en archivo: {len(existing_ids)}")