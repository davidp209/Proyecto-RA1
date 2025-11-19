import json
import time
import requests
import pandas as pd
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List

# --- CONFIGURACIÓN ---
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

# Rutas (Detecta automáticamente la carpeta del proyecto)
BASE_DIR = Path(__file__).resolve().parent.parent / "landing"
INPUT_FILE = BASE_DIR / "goodreads_books.json"
OUTPUT_FILE = BASE_DIR / "googlebooks_books.csv"

GOOGLE_API_URL = "https://www.googleapis.com/books/v1/volumes"
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

class GoogleBooksEnricher:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update(HEADERS)
        BASE_DIR.mkdir(parents=True, exist_ok=True)

    def load_data(self) -> List[Dict]:
        """Carga datos soportando tanto JSON Array como JSON Lines."""
        if not INPUT_FILE.exists():
            logger.error(f"No existe el archivo: {INPUT_FILE}")
            return []

        logger.info(f"Cargando: {INPUT_FILE}")
        data = []
        
        # Intento 1: Leer línea a línea (JSON Lines - Más común en scraping)
        try:
            with open(INPUT_FILE, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        try:
                            data.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue # Saltar líneas rotas
            if data: return data
        except Exception:
            pass # Si falla, probamos el método estándar

        # Intento 2: Leer todo el archivo como un bloque JSON
        try:
            with open(INPUT_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data
        except json.JSONDecodeError:
            logger.error("El archivo JSON no tiene un formato válido.")
            return []

    def _clean_str(self, text: str) -> str:
        """Limpia cadenas para evitar errores en búsquedas."""
        if not text: return ""
        return text.replace('"', '').replace("'", "").replace("“", "").replace("”", "").strip()

    def search_api(self, query: str, retries: int = 2) -> Optional[Dict]:
        """Consulta la API con manejo de Rate Limit."""
        params = {"q": query, "maxResults": 1, "printType": "books", "country": "ES"}
        
        for attempt in range(retries + 1):
            try:
                response = self.session.get(GOOGLE_API_URL, params=params, timeout=10)
                
                if response.status_code == 429: # Rate limit
                    wait = 2 ** attempt
                    logger.warning(f"Rate limit. Esperando {wait}s...")
                    time.sleep(wait)
                    continue
                
                if response.status_code == 200:
                    data = response.json()
                    if "items" in data:
                        return data["items"][0]
                    return None # No resultados
                
            except requests.RequestException as e:
                logger.error(f"Error de red: {e}")
        
        return None

    def find_book(self, book: Dict) -> Optional[Dict]:
        """
        Estrategia de búsqueda en cascada (Fallback).
        Prioridad: ISBN13 -> ISBN10 -> Título+Autor -> Título
        """
        

        title = self._clean_str(book.get("title", ""))
        isbn13 = book.get("isbn13")
        isbn10 = book.get("isbn")
        
        authors = book.get("authors", [])
        author = self._clean_str(authors[0]) if authors else ""

        # Definir estrategias
        strategies = []
        if isbn13: strategies.append(f"isbn:{isbn13}")
        if isbn10: strategies.append(f"isbn:{isbn10}")
        if title and author: strategies.append(f'intitle:"{title}" inauthor:"{author}"')
        if title: strategies.append(f'intitle:"{title}"')

        for query in strategies:
            result = self.search_api(query)
            if result:
                logger.info(f"Encontrado por: {query}")
                return result
        
        logger.warning(f"NO ENCONTRADO: {title}")
        return None

    def extract_fields(self, gr_id: str, item: Dict) -> Dict[str, Any]:
        """Extrae y limpia los datos finales."""
        vol = item.get("volumeInfo", {})
        sale = item.get("saleInfo", {})
        
        # Identificadores
        ids = {x.get("type"): x.get("identifier") for x in vol.get("industryIdentifiers", [])}

        # Precio
        price = sale.get("listPrice") or sale.get("retailPrice")
        amount = price.get("amount") if price else None
        currency = price.get("currencyCode") if price else None
        
        # Formateo de listas con " | " para seguridad en CSV
        authors_str = " | ".join(vol.get("authors", [])) if vol.get("authors") else None
        cats_str = " | ".join(vol.get("categories", [])) if vol.get("categories") else None

        return {
            "gb_id": item.get("id"),
            "title": vol.get("title"),
            "subtitle": vol.get("subtitle"),
            "authors": authors_str,
            "publisher": vol.get("publisher"),
            "pub_date": vol.get("publishedDate"),
            "language": vol.get("language"),
            "categories": cats_str,
            "isbn13": ids.get("ISBN_13"),
            "isbn10": ids.get("ISBN_10"),
            "price_amount": amount,
            "price_currency": currency
        }

    def run(self):
        books = self.load_data()
        if not books: return

        logger.info(f"Procesando {len(books)} libros...")
        enriched_data = []

        for i, book in enumerate(books, 1):
            logger.info(f"[{i}/{len(books)}] Buscando: {book.get('title')}")
            
            google_item = self.find_book(book)
            time.sleep(0.5) # Respetar API

            if google_item:
                parsed = self.extract_fields(book.get("id"), google_item)
                enriched_data.append(parsed)
            else:
                # Guardar registro vacío para trazabilidad
                enriched_data.append({
                    "gb_id": None,
                    "title": book.get("title"),
                    "authors": None
                })

        # Guardar CSV
        if enriched_data:
            df = pd.DataFrame(enriched_data)
            
            # Ordenar columnas
            cols = [
                "gb_id", "title", "subtitle", "authors", "publisher", 
                "pub_date", "language", "categories", "isbn13", 
                "isbn10", "price_amount", "price_currency"
            ]
            # Asegurar que existan las columnas
            df = df.reindex(columns=cols)
            
            # Guardar compatible con Excel Español
            df.to_csv(OUTPUT_FILE, index=False, sep=";", encoding="utf-8-sig")
            logger.info(f"Archivo generado exitosamente: {OUTPUT_FILE}")

if __name__ == "__main__":
    enricher = GoogleBooksEnricher()
    enricher.run()