# RAG-Vorlesung: Retrieval Augmented Generation mit DuckDB

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/abuckenhofer/rag-vorlesung/blob/main/RAG_DuckDB_Minimal.ipynb)

Begleitmaterial zur Ringvorlesung "Retrieval Augmented Generation" von Andreas Buckenhofer.

Das Notebook baut eine vollständige RAG-Anwendung fast nur mit SQL: Wikipedia-Artikel laden, in Chunks zerlegen, Embeddings berechnen, semantisch suchen, mit einem LLM antworten und das Ergebnis messen. Python wird nur in einer Zelle gebraucht, um das Embedding-Modell und das LLM als SQL-Funktionen in DuckDB zu registrieren.

## Schnellstart in Google Colab

1. Oben auf **Open in Colab** klicken.
2. Im Menü **Laufzeit > Alle ausführen** wählen.
3. Fertig. Es wird weder eine Installation noch ein API-Key benötigt.

Der erste Lauf dauert einige Minuten, weil das Embedding-Modell (ca. 400 MB) geladen und die Embeddings berechnet werden.

## LLM-Zugang

Der Schalter `LLM_ZUGANG` in der ersten Code-Zelle legt fest, wie das Notebook das LLM aufruft:

| `LLM_ZUGANG` | Was wird gebraucht? | Läuft wo? |
|---|---|---|
| `"colab"` (Standard) | Nichts. Gemini ist in Colab eingebaut, mit monatlichem Kontingent. | nur in Colab |
| `"account"` | Eigener kostenloser Gemini-API-Key | in Colab und lokal |

### Eigenen Gemini-API-Key einrichten (optional)

Der Key ist kostenlos und braucht keine Kreditkarte.

1. Mit einem **privaten Google-Konto** auf https://aistudio.google.com/apikey anmelden. Hochschulkonten (Google Workspace for Education) sperren den Zugriff häufig.
2. **API-Schlüssel erstellen** wählen und den Key kopieren.
3. In Colab links auf das Schlüsselsymbol klicken, ein Secret mit dem Namen `GEMINI` anlegen, den Key einfügen und den Notebook-Zugriff aktivieren. Lokal stattdessen die Umgebungsvariable `GEMINI_API_KEY` setzen.
4. Im Notebook `LLM_ZUGANG = "account"` setzen.

Die kostenlosen Kontingente sind knapp. Das voreingestellte Modell `gemini-3.5-flash-lite` hat deutlich mehr Anfragen pro Tag frei als die größeren Flash-Modelle. Inhalte aus dem kostenlosen Zugang darf Google zur Verbesserung seiner Produkte nutzen, deshalb keine vertraulichen Dokumente verwenden.

## Lokal ausführen

Mit [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/abuckenhofer/rag-vorlesung.git
cd rag-vorlesung
uv init --bare
uv add duckdb sentence-transformers google-genai jupyterlab
export GEMINI_API_KEY=...   # Windows: set GEMINI_API_KEY=...
uv run jupyter lab
```

Lokal steht Colab AI nicht zur Verfügung, daher im Notebook `LLM_ZUGANG = "account"` setzen.

## Aufbau des Notebooks

| Schritt | Teil der RAG-Architektur | Inhalt |
|---|---|---|
| 1 Setup | | Konfiguration, Installation, Registrierung von `embed()` und `llm()` |
| 2 Daten laden | Index Pipeline | Wikipedia-Artikel per `read_json` als reinen Text |
| 3 Chunking | Index Pipeline | Ein Absatz ergibt einen Chunk, mit Überschrift als Kontext |
| 4 Embeddings | Index Pipeline | `all-mpnet-base-v2`, 768 Dimensionen |
| 5 Semantische Suche | Vektorspeicher | Makro `suche(frage, k)` mit Kosinus-Distanz |
| 6 RAG | RAG Pipeline | Makros `prompt()` und `rag()`, Antwort mit Quellenangabe |
| 7 Evaluation | | Rang, Recall@k und MRR für das Retrieval, Diagnose der Antworten |

Die Datenbank liegt nur im Arbeitsspeicher. Jede Zelle außer dem Setup lässt sich beliebig oft wiederholen. Nach einem erneuten Setup die folgenden Zellen noch einmal ausführen.

## Grenzen des Minimalbeispiels

- **Chunking:** Die Trennung nach Absätzen ist bewusst einfach. Sehr lange Absätze schneidet das Embedding-Modell nach 384 Tokens ab, eine Abfrage in Schritt 3 zeigt betroffene Chunks.
- **Suche:** Die Suche ist exakt und vergleicht die Frage mit jedem Chunk. Für große Datenmengen bietet die DuckDB-Extension `vss` einen HNSW-Index.
- **Evaluation:** Das Testset prüft über Stichwörter, ob der passende Chunk gefunden wurde. Das ist eine Heuristik und irrt in beide Richtungen. Ob eine Antwort inhaltlich stimmt, prüft sie nicht.
- **Testset:** Fünf Fragen zeigen das Prinzip, für belastbare Aussagen braucht es 20 bis 50 Fragen, formuliert wie echte Nutzer fragen.

## Ideen zum Weitermachen

- Eigene Dokumente laden, z.B. mit `read_text('*.md')`
- Chunking variieren und Recall@k vergleichen
- Hybrid Search: Volltextsuche mit der Extension `fts` und semantische Suche per Reciprocal Rank Fusion kombinieren
- Ein anderes Embedding-Modell testen, z.B. ein mehrsprachiges für deutsche Texte

## Weiterführend

- [tdwi2025](https://github.com/abuckenhofer/tdwi2025): Workshop-Notebook mit LangChain, FAISS, DuckDB und PostgreSQL pgvector
- [Vector Database: What, Why and How](https://buckenhofer.com/2024/05/vector-database-what-why-and-how/)
- [DOAG Datenbank-Kolumne: Was ist eine Vektordatenbank?](https://www.doag.org/de/home/news/doag-datenbank-kolumne-was-ist-eine-vektordatenbank/)

## Lizenz

MIT, siehe [LICENSE](LICENSE).
