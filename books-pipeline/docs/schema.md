# Data Dictionary: dim_book

**Archivo fuente:** `landing/dim_book.parquet`  
**Descripción:** Tabla dimensional consolidada que unifica datos de Goodreads y Google Books. Prioriza la fuente con mejor calidad de datos mediante `merge_books_pipeline.py`. Contiene información normalizada y enriquecida para análisis y reporting.

---

## Campos

| Campo | Tipo | Nullable | Ejemplo | Reglas y Lógica de Negocio |
| :--- | :--- | :---: | :--- | :--- |
| **canonical_id** | STRING | No | `"a1b2c3d4..."` | **PK**. Identificador único estable. Si existe ISBN13, se usa como ID. Si no, se genera SHA1 de `title_normalized + first_author + publisher + pub_year`. |
| **isbn13** | STRING | Sí | `"9781234567890"` | ISBN13 normalizado a string. Se usa como ID preferente si existe. |
| **isbn10** | STRING | Sí | `"1234567890"` | ISBN10 legacy. Puede ser nulo. |
| **title** | STRING | Sí | `"Clean Code"` | Título original. Se elige según fuente preferente y completitud de datos. |
| **title_normalized** | STRING | Sí | `"cleancode"` | Título en minúsculas, sin puntuación ni espacios extra. Usado para matching heurístico. |
| **authors** | STRING | Sí | `"Robert C. Martin | Dean Wampler"` | Lista de autores normalizada y separada por `|`. Se eliminan duplicados y strings vacíos. |
| **first_author** | STRING | Sí | `"Robert C. Martin"` | Primer autor de la lista, usado para índices de búsqueda rápida. Nullable si no hay autores. |
| **publisher** | STRING | Sí | `"Prentice Hall"` | Editorial normalizada. Se elige según fuente preferente. |
| **pub_date** | STRING | Sí | `"2008-08-01"` | Fecha de publicación en formato ISO 8601 (YYYY-MM-DD, YYYY-MM o YYYY). Se normaliza de ambas fuentes. |
| **pub_year** | INT | Sí | `2008` | Año extraído de `pub_date`. Usado para particionamiento o filtrado. |
| **language** | STRING | Sí | `"eng"` | Código ISO del idioma. Se prioriza Goodreads, si no existe, se toma Google Books. |
| **categories** | STRING | Sí | `"Technology | Programming"` | Categorías unificadas y normalizadas separadas por `|`. Se eliminan duplicados y strings vacíos. |
| **num_pages** | INT | Sí | `464` | Número de páginas. Prioriza Goodreads por defecto. |
| **format** | STRING | Sí | `"Paperback"` | Formato físico/digital del libro. Se elige según fuente preferente. |
| **description** | STRING | Sí | `"Even bad code..."` | Sinopsis o descripción. Se elige según fuente preferente. |
| **rating_value** | FLOAT | Sí | `4.4` | Puntuación promedio de Goodreads. |
| **rating_count** | INT | Sí | `2500` | Cantidad de votos en Goodreads. |
| **price_amount** | FLOAT | Sí | `35.50` | Precio numérico. Prioriza Google Books si existe. |
| **price_currency** | STRING | Sí | `"USD"` | Moneda normalizada a ISO 4217 (USD, EUR, GBP). Símbolos `$`, `€`, `£` se convierten automáticamente. |
| **source_preference** | STRING | No | `"goodreads"` | Fuente que ganó la prioridad (`goodreads` o `google`). Determina qué valores se toman en caso de conflicto. |
| **most_complete_url** | STRING | Sí | `"https://goodreads.com/book/show/12345"` | URL del recurso más completo según heurística de fuente preferida. |
| **ingestion_date_goodreads** | TIMESTAMP | Sí | `"2023-10-27T12:34:56Z"` | Fecha de extracción original desde Goodreads. |
| **ingestion_date_google** | TIMESTAMP | Sí | `"2023-10-28T08:12:00Z"` | Fecha de extracción original desde Google Books (puede ser nulo). |

---

## Glosario de campos multivalor

| Campo | Ejemplo | Significado | Notas |
| :--- | :--- | :--- | :--- |
| **authors** | `"Robert C. Martin | Dean Wampler"` | Lista de autores del libro | Se normalizan, eliminan duplicados y strings vacíos. Primer autor → `first_author`. Separador: `|`. |
| **categories** | `"Technology | Programming"` | Lista de categorías del libro | Se normalizan y eliminan duplicados/strings vacíos. Separador: `|`. |

**Reglas generales para campos multivalor:**
1. Separador siempre es `|` sin espacios antes/después.  
2. Elementos vacíos o `None` se eliminan.  
3. Orden: primero datos de Goodreads, luego Google Books si hay elementos nuevos.  
4. Uso en consultas: filtrar con `LIKE '%valor%'` o separar la cadena usando `|`.

---

## Notas adicionales

1. **Normalización**
   - `title_normalized`, `authors` y `categories` se limpian de espacios extra y caracteres especiales.
   - Garantiza consistencia para matching y análisis.

2. **Canonical ID**
   - Siempre existe y asegura unicidad.
   - SHA1 se genera concatenando `title_normalized + first_author + publisher + pub_year` si no hay ISBN13.

3. **Merge de fuentes**
   - Se prioriza la fuente con mayor **score de completitud** según heurística definida en `merge_books_pipeline.py`.
   - `source_preference` indica la fuente elegida.

4. **Métricas de calidad**
   - Se generan en `quality_metrics.json`:
     - `rows_input_goodreads`
     - `rows_output`
     - `matched_with_google`

5. **Fechas**
   - `pub_date` puede ser solo año, año-mes o completo.
   - `ingestion_date_*` está en UTC ISO 8601.

6. **URLs**
   - `most_complete_url` refleja la fuente más confiable según la heurística del pipeline.
