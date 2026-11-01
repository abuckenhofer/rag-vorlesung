# RAG-Vorlesung: Retrieval Augmented Generation mit DuckDB

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/abuckenhofer/rag-vorlesung/blob/main/RAG_DuckDB_Minimal.ipynb)

Begleitmaterial zur Ringvorlesung "Retrieval Augmented Generation" von Andreas Buckenhofer.

Das Notebook baut eine vollständige RAG-Anwendung fast nur mit SQL: Wikipedia-Artikel laden, in Chunks zerlegen, Embeddings berechnen, semantisch suchen, mit einem LLM antworten und das Ergebnis messen. Python wird nur in einer Zelle gebraucht, um das Embedding-Modell und das LLM als SQL-Funktionen in DuckDB zu registrieren. Alle anderen Zellen beginnen mit `%%sql` und enthalten reines SQL, ausgeführt über die Erweiterung [jupysql](https://jupysql.ploomber.io).

Nach den einzelnen Schritten folgen 17 kurze Übungen zu Tabellen, Tokens, Chunking, Ähnlichkeit und RAG, teils mit Lücken `___` zum Ergänzen. Die Lösungen sind eingeklappt. Der Hauptteil hängt von keiner Übung ab, Übungen lassen sich also beliebig überspringen. Umgekehrt laufen auch ungelöste Übungszellen fehlerfrei durch.

## Schnellstart in Google Colab

1. Oben auf **Open in Colab** klicken.
2. Einen OpenRouter-Key als Secret `OPENROUTER_API_KEY` anlegen (siehe unten) oder ohne Key in der ersten Code-Zelle `LLM_ZUGANG = "colab"` setzen.
3. Im Menü **Laufzeit > Alle ausführen** wählen. Eine Installation ist nicht nötig.

Der erste Lauf dauert einige Minuten, weil das Embedding-Modell (ca. 400 MB) geladen und die Embeddings berechnet werden.

## LLM-Zugang

Der Schalter `LLM_ZUGANG` in der ersten Code-Zelle legt fest, wie das Notebook das LLM aufruft:

| `LLM_ZUGANG` | Was wird gebraucht? | Läuft wo? |
|---|---|---|
| `"colab"` | Nichts. Gemini ist in Colab eingebaut, mit monatlichem Kontingent. | nur in Colab |
| `"account"` | Eigener kostenloser Gemini-API-Key | in Colab und lokal |
| `"openrouter"` (Standard) | Eigener OpenRouter-Key, beliebiges Modell über openrouter.ai. Voreingestellt ist `z-ai/glm-5.1`. | in Colab und lokal |

### Eigenen Gemini-API-Key einrichten (optional)

Der Key ist kostenlos und braucht keine Kreditkarte.

1. Mit einem **privaten Google-Konto** auf https://aistudio.google.com/apikey anmelden. Hochschulkonten (Google Workspace for Education) sperren den Zugriff häufig.
2. **API-Schlüssel erstellen** wählen und den Key kopieren.
3. In Colab links auf das Schlüsselsymbol klicken, ein Secret mit dem Namen `GEMINI` anlegen, den Key einfügen und den Notebook-Zugriff aktivieren. Lokal stattdessen die Umgebungsvariable `GEMINI_API_KEY` setzen.
4. Im Notebook `LLM_ZUGANG = "account"` setzen.

Die kostenlosen Kontingente sind knapp. Das voreingestellte Modell `gemini-3.5-flash-lite` hat deutlich mehr Anfragen pro Tag frei als die größeren Flash-Modelle. Inhalte aus dem kostenlosen Zugang darf Google zur Verbesserung seiner Produkte nutzen, deshalb keine vertraulichen Dokumente verwenden.

### OpenRouter einrichten (optional)

OpenRouter bietet viele Modelle über eine einheitliche API. Das voreingestellte Modell `z-ai/glm-5.1` ist kostenpflichtig und braucht Guthaben auf dem OpenRouter-Konto. Freie Modelle (Endung `:free`, z.B. `meta-llama/llama-3.3-70b-instruct:free`) brauchen keine Kreditkarte, sind aber auf 50 Anfragen am Tag begrenzt. Ein Durchlauf des Notebooks braucht 11 Anfragen, mit Anhang 19.

1. Auf https://openrouter.ai anmelden und unter https://openrouter.ai/keys einen Key erstellen.
2. In Colab ein Secret `OPENROUTER_API_KEY` anlegen, lokal die gleichnamige Umgebungsvariable setzen.
3. Bei Bedarf `OPENROUTER_MODELL` ändern, z.B. auf ein freies Modell aus https://openrouter.ai/models.

Die Setup-Zelle prüft den Key sofort und zeigt, wie viele freie Anfragen heute noch übrig sind. Häufige Fehler:

- **401 Unauthorized:** Der Key ist ungültig, gelöscht oder unvollständig kopiert. Unter https://openrouter.ai/keys einen neuen Key erstellen und vollständig in das Secret kopieren.
- **Secret nicht gefunden:** Der Name des Secrets muss exakt `OPENROUTER_API_KEY` lauten, und der Schalter für den Notebook-Zugriff muss aktiv sein.
- **402 Payment Required:** Für das kostenpflichtige Modell fehlt Guthaben. Guthaben aufladen oder ein `:free`-Modell wählen.
- **429 Too Many Requests:** Das Tageslimit von 50 Anfragen ist erreicht, oder das freie Modell ist gerade überlastet. Später erneut versuchen oder ein anderes `:free`-Modell wählen.

## Lokal ausführen

Mit [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/abuckenhofer/rag-vorlesung.git
cd rag-vorlesung
uv init --bare
uv add duckdb jupysql sentence-transformers google-genai requests jupyterlab
export GEMINI_API_KEY=...       # oder OPENROUTER_API_KEY; Windows: set ...
uv run jupyter lab
```

Lokal steht Colab AI nicht zur Verfügung, daher im Notebook `LLM_ZUGANG = "account"` oder `"openrouter"` setzen.

## Aufbau des Notebooks

| Schritt | Teil der RAG-Architektur | Inhalt |
|---|---|---|
| 1 Setup | | Konfiguration, Installation, Registrierung von `embed()`, `llm()` und `tokens()`, Verbindung von jupysql mit DuckDB |
| 2 Daten laden | Index Pipeline | Wikipedia-Artikel per `read_json` als reinen Text |
| 3 Chunking | Index Pipeline | Ein Absatz ergibt einen Chunk, mit Überschrift als Kontext |
| 4 Embeddings | Index Pipeline | `all-mpnet-base-v2`, 768 Dimensionen |
| 5 Semantische Suche | Vektorspeicher | Makro `suche(frage, k)` mit Kosinus-Distanz |
| 6 RAG | RAG Pipeline | Makros `prompt()` und `rag()`, Antwort mit Quellenangabe |
| 7 Evaluation | | Rang, Recall@k und MRR für das Retrieval, Diagnose der Antworten |
| Anhang (optional) | | A Hybrid Search, B Reranking, C Query Rewriting und HyDE, D Metadaten-Filter, E HNSW-Index, F LLM-as-a-Judge |

| Schritt | Übungen |
|---|---|
| 3 Chunking | Katalog mit `information_schema`, Chunks je Artikel, `SUMMARIZE`, Tokenizer, Token-Grenze des Modells, feste Länge und Überlappung |
| 4 Embeddings | Speicherbedarf und Quantisierung, Skalarprodukt und Distanzmaße, Ähnlichkeit von Wörtern |
| 5 Semantische Suche | Deutsche Frage, Volltext gegen semantische Suche, Tippfehler |
| 6 RAG | Wahl von k, eigener System-Prompt, Prompt Injection |
| 7 Evaluation | Eigene Testfrage, Recall bei Chunks mit fester Länge |

Der Anhang ist optional. Jeder Abschnitt setzt nur den Hauptteil voraus und lässt sich einzeln ausführen. Abschnitt A lädt die DuckDB-Extension `fts`, Abschnitt B das Modell `cross-encoder/ms-marco-MiniLM-L-6-v2` (ca. 90 MB), Abschnitt E die Extension `vss`. Abschnitt C braucht 2, Abschnitt F 6 LLM-Anfragen. Wer nur den Hauptteil ausführen will, wählt in der Zelle *Anhang* den Menüpunkt *Laufzeit > Vorherige ausführen*.

Die Datenbank liegt nur im Arbeitsspeicher. Jede Zelle außer dem Setup lässt sich beliebig oft wiederholen. Nach einem erneuten Setup die folgenden Zellen noch einmal ausführen.

## Grenzen des Minimalbeispiels

- **Chunking:** Die Trennung nach Absätzen ist bewusst einfach. Sehr lange Absätze schneidet das Embedding-Modell nach 384 Tokens ab, eine Abfrage in Schritt 3 zeigt betroffene Chunks.
- **Suche:** Die Suche ist exakt und vergleicht die Frage mit jedem Chunk. Für große Datenmengen bietet die DuckDB-Extension `vss` einen HNSW-Index.
- **Evaluation:** Das Testset prüft über Stichwörter, ob der passende Chunk gefunden wurde. Das ist eine Heuristik und irrt in beide Richtungen. Ob eine Antwort inhaltlich stimmt, prüft sie nicht.
- **Testset:** Fünf Fragen zeigen das Prinzip, für belastbare Aussagen braucht es 20 bis 50 Fragen, formuliert wie echte Nutzer fragen.

## Ideen zum Weitermachen

- Eigene Dokumente laden, z.B. mit `read_text('*.md')`
- Chunking nach Sätzen oder mit einem Text-Splitter wie in LangChain und Recall@k vergleichen
- Hybrid Search aus dem Anhang mit Reranking kombinieren und auf einem größeren Testset messen
- Ein anderes Embedding-Modell testen, z.B. ein mehrsprachiges für deutsche Texte

## Weiterführend

- [tdwi2025](https://github.com/abuckenhofer/tdwi2025): Workshop-Notebook mit LangChain, FAISS, DuckDB und PostgreSQL pgvector
- [Vector Database: What, Why and How](https://buckenhofer.com/2024/05/vector-database-what-why-and-how/)
- [DOAG Datenbank-Kolumne: Was ist eine Vektordatenbank?](https://www.doag.org/de/home/news/doag-datenbank-kolumne-was-ist-eine-vektordatenbank/)

