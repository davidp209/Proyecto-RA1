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
    isbn13: Optional[str] = None
    title: Optional[str] = None
    authors: List[str] = field(default_factory=list)
    
    publisher: Optional[str] = None  # <--- OBJETIVO A RELLENAR
    
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

# --- 3. FUNCIONES DE LIMPIEZA Y EXTRACCIÓN ---

def clean_text_deep(text):
    """Limpia descripciones HTML"""
    if not text: return None
    text = text.replace("<br>", "\n").replace("<br />", "\n")
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text(separator="\n").strip()

def extract_pages_from_html(soup):
    """Busca número de páginas"""
    try:
        p_tag = soup.find("p", {"data-testid": "pagesFormat"})
        if p_tag:
            match = re.search(r'(\d+)', p_tag.get_text())
            if match: return int(match.group(1))
    except:
        pass
    return None

def extract_publisher_info(soup, json_publisher):
    """
    Intenta sacar el Publisher.
    1. Del JSON si existe.
    2. Del texto 'Published ... by O'Reilly' en el HTML.
    """
    # 1. Intento fácil: JSON
    if json_publisher:
        return json_publisher

    # 2. Intento difícil: Buscar en la frase de publicación HTML
    # GoodReads pone: "First published January 1, 2020 by O'Reilly Media"
    try:
        pub_tag = soup.find("p", {"data-testid": "publicationInfo"})
        if pub_tag:
            text = pub_tag.get_text()
            # Buscamos la palabra mágica " by "
            if " by " in text:
                # Cortamos el texto. [1] es lo que hay después del " by "
                parts = text.split(" by ")
                if len(parts) > 1:
                    publisher_raw = parts[-1]
                    # A veces hay basura después, intentamos limpiarlo
                    return publisher_raw.strip()
    except:
        pass
    
    return None

# --- 4. SCRAPER PRINCIPAL ---

def get_book_details(book_id):
    url = BASE_BOOK_URL + str(book_id)
    try:
        resp = requests.get(url, headers=HEADERS)
        resp.raise_for_status()
    except Exception:
        return None

    soup = BeautifulSoup(resp.text, "html.parser")
    
    # --- A. JSON-LD ---
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

    # --- B. EXTRACCIÓN ---
    
    # Título y Portada
    title = book_json.get('name')
    if not title:
        meta_title = soup.find("meta", property="og:title")
        if meta_title: title = meta_title.get("content")

    # Descripción
    raw_desc = book_json.get('description')
    if not raw_desc:
        d_div = soup.find("div", {"data-testid": "description"})
        if d_div: raw_desc = d_div.get_text(separator="\n")
    final_desc = clean_text_deep(raw_desc)

    # ISBN
    isbn13 = book_json.get('isbn13') or book_json.get('isbn')

    # Autores
    authors_list = []
    raw_author = book_json.get('author')
    if raw_author:
        if isinstance(raw_author, list):
            authors_list = [a.get('name') for a in raw_author if isinstance(a, dict)]
        elif isinstance(raw_author, dict):
             if raw_author.get('name'): authors_list = [raw_author.get('name')]
    if not authors_list:
        for sp in soup.find_all("span", class_="ContributorLink__name"):
            if sp.text not in authors_list: authors_list.append(sp.text)

    # Categorías
    cats_list = []
    for link in soup.find_all("a", href=re.compile(r'/genres/')):
        g = link.get_text(strip=True)
        if g and len(g) > 2 and g not in cats_list: cats_list.append(g)
    cats_list = list(set(cats_list))[:5]

    # Fecha
    p_date = book_json.get('datePublished')
    if not p_date:
        pub_tag = soup.find("p", {"data-testid": "publicationInfo"})
        if pub_tag: 
            # Extraer fecha del texto "Published October 2020 by..."
            # Esto es complejo, intentamos coger la fecha del JSON o dejar el texto entero temporalmente
            # Pero para "pub_date" solemos querer YYYY-MM-DD. Si falla, dejamos None o el texto.
            p_date = pub_tag.get_text().split(" by ")[0].replace("First published ", "").replace("Published ", "")

    # --- LOGICA PUBLISHER (NUEVA) ---
    # Sacamos el publisher del JSON (si hay) o del HTML
    json_pub_name = book_json.get('publisher', {}).get('name') if isinstance(book_json.get('publisher'), dict) else None
    final_publisher = extract_publisher_info(soup, json_pub_name)

    return BookData(
        id=str(book_id),
        isbn13=str(isbn13) if isbn13 else None,
        title=title,
        authors=authors_list,
        
        publisher=final_publisher, # <--- DATO ASIGNADO
        
        pub_date=p_date,
        language=book_json.get('inLanguage'),
        categories=cats_list,
        desc=final_desc,
        num_pages=book_json.get('numberOfPages') or extract_pages_from_html(soup),
        format=book_json.get('bookFormat'),
        rating_value=float(book_json.get('aggregateRating', {}).get('ratingValue', 0)) if book_json.get('aggregateRating') else None,
        rating_count=int(book_json.get('aggregateRating', {}).get('ratingCount', 0)) if book_json.get('aggregateRating') else None,
        url=url,
        ingestion_date=datetime.now().isoformat()
    )

def get_book_ids_from_search(query, max_pages=1):
    found_ids = []
    print(f"--- Buscando '{query}' ---")
    for page in range(1, max_pages + 1):
        params = {'q': query, 'page': page, 'search_type': 'books'}
        try:
            resp = requests.get(SEARCH_URL, headers=HEADERS, params=params)
            soup = BeautifulSoup(resp.text, "html.parser")
            for link in soup.find_all("a", class_="bookTitle"):
                match = re.search(r'/show/(\d+)', link.get('href'))
                if match and match.group(1) not in found_ids:
                    found_ids.append(match.group(1))
        except: break
    return found_ids

if __name__ == "__main__":
    TERMINO = "Data Science"
    ids = get_book_ids_from_search(TERMINO, max_pages=1)
    ids_to_scrape = ids[:5] # Solo 5 para probar
    
    print(f"\n>>> Procesando {len(ids_to_scrape)} libros...\n")

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for i, bid in enumerate(ids_to_scrape):
            bk = get_book_details(bid)
            if bk and bk.title:
                f.write(json.dumps(asdict(bk), ensure_ascii=False) + "\n")
                
                # Debug para que veas si funciona el Publisher
                pub_str = bk.publisher if bk.publisher else "NO ENCONTRADO"
                print(f"[{i+1}] {bk.title[:20]}... | Pub: {pub_str}")
            else:
                print(f"[{i+1}] Error")
            
            time.sleep(random.uniform(1.0, 2.0))

    print(f"\n>>> FINALIZADO: {OUTPUT_FILE}")