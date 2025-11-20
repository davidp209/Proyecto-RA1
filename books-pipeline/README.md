# Books Pipeline: Ingesta y Unificación de Datos de Libros (Proyecto RA1)

Este proyecto implementa un pipeline ETL (Extract, Transform, Load) diseñado para construir un catálogo de libros robusto y normalizado (`dim_book`). El sistema extrae información social y descriptiva de **Goodreads**, la enriquece con datos comerciales (precios, ISBNs precisos) de **Google Books API**, y unifica ambas fuentes mediante un algoritmo de resolución de entidades.

## 📋 Descripción del Flujo

El pipeline opera en tres etapas secuenciales:

1. **Scraping (Goodreads):** Obtención de metadatos, ratings y descripciones mediante scraping híbrido (HTML parsing + JSON-LD).
2. **Enriquecimiento (Google Books):** Búsqueda de libros coincidentes vía API para completar ISBNs faltantes y precios.
3. **Normalización y Merge:** Fusión de ambas fuentes priorizando la calidad del dato, limpieza de strings y generación de una tabla dimensional final.

## 📂 Estructura

```bash
BOOKS-PIPELINE/
│
├── 📂 docs/
│   ├── quality_metrics.json → métricas de calidad (filas, cruces, totales)
│   └── schema.md → esquema documentado de la tabla final
│
├── 📂 landing/
│   ├── goodreads_books.json → fuente bruta de Goodreads (JSONL)
│   └── googlebooks_books.csv → datos enriquecidos desde Google Books
│
├── 📂 src/
│   ├── scraper_goodreads.py  → extracción (scraping) desde Goodreads
│   ├── enrich_googlebooks.py → enriquecimiento usando Google Books 
│   └── integrate_pipeline.py →  integración / merge / normalización
│
├── 📂 standard/
│   ├── dim_book.parquet → tabla maestra de libros (modelo canónico)
│   └── book_source_detail.parquet → detalle de trazabilidad de fuentes
│
└── requirements.txt → dependencias del proyecto

```

## 🚀 Cómo Ejecutar

- Python 3.8+ (especificar versión exacta si aplica)

### 1️⃣ Instalar dependencias:

```bash
pip install -r requirements.txt
```

### 2️⃣ Activar entorno virtual

```bash
venv\Scripts\activate
```

## 🚀 Ejecución paso a paso

### 1️⃣ Scrapear datos de Goodreads

Obtiene metadatos, descripciones y ratings mediante scraping.

```bash
python src/scraper_goodreads.py
```

### 2️⃣ Enriquecer datos usando Google Books API

Busca precios e ISBNs faltantes mediante coincidencia difusa (fuzzy matching).

```bash
python src/enrich_googlebooks.py
```

### 3️⃣ Integrar y normalizar en el modelo canónico

Ejecuta la lógica de fusión ("Survivor Value"), limpieza y deduplicación.

```bash
python src/integrate_pipeline.py
```

## 🗂 Metadatos y configuraciones

* **Separador CSV:** `;`
* **Codificación:** UTF-8
* **Selectores/UA:** No aplica, se usan archivos locales; ingestión proviene de `ingestion_date` de cada fuente.
* **Normalización de datos:**

  * Autores y categorías → unidos con `|`.
  * Títulos → normalizados sin puntuación ni espacios extra (`title_normalized`).
  * Precios → normalizados a ISO 4217 (`USD`, `EUR`, `GBP`).

## 🔑 Decisiones clave

* **Unificación de fuentes**

  * Prioridad basada en **score de completitud** entre Goodreads y Google Books.
  * `source_preference` indica la fuente elegida.
* **Canonical ID**

  * Si existe ISBN13 → se usa como PK.
  * Si no → SHA1 de `title_normalized + first_author + publisher + pub_year`.
* **Matching heurístico**

  * Primero por ID explícito (`id` de Goodreads vs `gb_id` de Google Books).
  * Luego por ISBN13.
  * Finalmente por combinación `title_normalized + first_author`.
* **Manejo de tipos mixtos**

  * Strings y números normalizados para evitar errores Parquet (`ArrowTypeError`).
  * Fechas convertidas a ISO 8601; nulos permitidos.
* **Fallback de guardado**

  * Si Parquet falla, se guarda en CSV con conversión a string para evitar pérdida de datos.

  ## 🗃 Esquema de `dim_book

[schema.md](./docs/schema.md)

## 📝 Notas adicionales

* Normalización garantiza consistencia para matching y análisis.
* Canonical ID asegura unicidad y estabilidad.
* Merge prioriza fuente con mayor completitud.
* Métricas de calidad (`quality_metrics.json`) ayudan a auditar la cobertura y coincidencia entre fuentes.
* Fechas `pub_date` y `ingestion_date_*` en formato UTC ISO 8601.

# 🛡 Idempotencia y Deduplicación en merge_books_pipeline

## 🔹 Idempotencia

El pipeline está diseñado para **ser idempotente**, es decir, **puede ejecutarse varias veces sin crear duplicados ni inconsistencias** en la tabla `dim_book`.

**Mecanismos que garantizan la idempotencia:**

1. **Canonical ID único**

   - Se genera un `canonical_id` para cada libro:
     - Si existe `isbn13` → se usa como ID.
     - Si no → SHA1 de `[title_normalized, first_author, publisher, pub_year]`.
   - Esto asegura que el mismo libro tenga siempre el mismo identificador, sin importar cuántas veces se ejecute el pipeline.
2. **Normalización consistente**

   - Títulos, autores, categorías y fechas se normalizan antes del merge:
     - `title_normalized`: minúsculas, sin puntuación ni espacios extra.
     - `authors`: lista única separada por `|`.
     - `categories`: lista única separada por `|`.
   - Evita que diferencias de formato generen duplicados.
3. **Merge determinista**

   - Matching en tres niveles:

     1. `gb_id` (ID de Google Books)
     2. `isbn13`
     3. Clave heurística `title_normalized + first_author`
4. **Salida reproducible**

   - Los archivos Parquet/CSV generados contienen siempre las mismas filas si los datos de entrada no cambian.
   - Las métricas de calidad (`quality_metrics.json`) reflejan los resultados de forma consistente.

---

## 🔹 Deduplicación

El pipeline incluye un paso explícito de **deduplicación** para eliminar registros repetidos en `dim_book`.

**Cómo se implementa:**

1. Se añade una columna temporal `_score`:
   ```python
   df_final["_score"] = df_final.notnull().sum(axis=1)
   ```
