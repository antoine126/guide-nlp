# guide-nlp

Code compagnon du livre **Le Guide du NLP — Le traitement du langage naturel avec
Python** (Antoine Pagneux).

Chaque fichier cité dans le livre se trouve ici, au chemin indiqué au-dessus du
listing (`src/guide_nlp/chapter04/dates.py`, etc.), complété pour s'exécuter tel
quel. Les extraits imprimés montrent l'essentiel ; ce dépôt contient le reste :
imports, chargement des données, fonction `main()` et, pour la plupart des
chapitres, le calcul des chiffres publiés.

## Installation

Il faut [uv](https://docs.astral.sh/uv/). Il installe Python 3.12 si besoin.

```sh
uv sync                    # groupes par défaut : dev + text (chapitres 1 à 7)
uv sync --all-groups       # tout le livre
uv run --all-groups python scripts/check_env.py
```

Les dépendances sont rangées par groupes (chapitre 2) :

| Groupe    | Contenu                                                  | Chapitres |
|-----------|----------------------------------------------------------|-----------|
| `dev`     | pytest, ruff, ty                                         | tous      |
| `text`    | regex, spaCy et ses modèles, pySBD, simplemma, lingua, datasketch, Presidio | 3 à 7, 10 |
| `web`     | httpx, BeautifulSoup, lxml, selectolax, trafilatura, PyMuPDF, pdfplumber, Scrapy, Playwright | 8 et 9 |
| `classic` | scikit-learn, SciPy, gensim, bm25s                       | 11 à 15   |
| `neural`  | PyTorch, Transformers, Sentence Transformers, FAISS, SetFit, BERTopic | 16 à 31, 42, 43 |
| `llm`     | ollama, openai, anthropic                                | 32 à 34   |
| `rag`     | LlamaIndex, networkx, igraph, leidenalg, LangGraph, MCP  | 35 à 41   |
| `audio`   | faster-whisper, jiwer, librosa, Piper                    | 45        |
| `serve`   | FastAPI, Uvicorn                                         | 46        |

PyTorch vient de l'index officiel `cu132` (CUDA 13.2), avec lequel les
chiffres du livre ont été mesurés. Adaptez le suffixe à votre pilote
(`cu126`, `cu128`, `cu130`…) dans `pyproject.toml` ; sur macOS, le paquet
standard utilise MPS.

Trois outils exigent des versions incompatibles avec cet environnement. Ils
ont chacun le leur, dans `envs/` :

```sh
uv sync --project envs/vision      # ColPali (ch. 43) : torch < 2.14
uv sync --project envs/pageindex   # PageIndex (ch. 40) : huggingface-hub 1.x
uv sync --project envs/docling     # docling (ch. 8)  : huggingface-hub 1.x
uv run --project envs/vision python -m guide_nlp.chapter43.colqwen_search
```

Le premier `playwright` du chapitre 9 demande `uv run playwright install chromium`.

## Lancer le code

Chaque module se lance par son nom ; sa docstring indique le groupe à
installer et ce qu'il affiche.

```sh
uv run python -m guide_nlp.chapter04.dates
uv run --group classic python -m guide_nlp.chapter13.bm25
uv run --group neural python -m guide_nlp.chapter28.bench
```

Les extraits courts sans chemin de fichier dans le livre sont regroupés, dans
l'ordre du texte, dans `chapterNN/snippets.py`. Les modules `experiments.py`
des chapitres 11 à 15 recalculent les tableaux de résultats de ces chapitres ;
ailleurs, le `main()` du module principal reproduit le résultat central.

Les chapitres 32 à 41, 44 et 47 interrogent un serveur
[Ollama](https://ollama.com) local (`http://127.0.0.1:11434`, réglable par
`GUIDE_NLP_OLLAMA_URL`). Modèles utilisés :

```sh
ollama pull qwen2.5:7b-instruct-q4_K_M   # le modèle principal
ollama pull qwen2.5vl:3b                 # chapitre 44
ollama pull smollm2:135m deepseek-r1:8b qwen3:8b
```

Les réponses du LLM sont mises en cache dans `data/interim/llm_cache.jsonl` :
relancer un script ne refait pas les appels.

## Organisation

```
guide-nlp/
├── pyproject.toml, uv.lock, .python-version
├── src/guide_nlp/
│   ├── config.py, runtime.py, models.py   réglages, périphérique, modèles figés (ch. 2)
│   ├── llm.py                             client Ollama typé, avec cache (ch. 32 à 47)
│   ├── rag.py                             la base RAG commune (ch. 37 à 41)
│   ├── text/                              normalisation, motifs, prétraitement (ch. 3 à 6)
│   ├── corpus/                            collecte, nettoyage, chargement des corpus
│   ├── chapter01/ … chapter47/            un module par chapitre
│   └── annexes/                           expressions régulières de l'annexe
├── data/
│   ├── raw/          données brutes (échantillons du ch. 8 ; le reste se télécharge)
│   ├── interim/      résultats intermédiaires, caches (non versionnés)
│   └── processed/    le corpus « Articles » et les jeux de questions
├── scripts/          check_env, construction du corpus, fichiers d'exemple
├── tests/            tests rapides (pytest)
└── envs/             les trois environnements séparés
```

## Données

Fournies dans `data/processed/` (voir `SOURCES.md`) :

- `articles.jsonl` et `articles.parquet` : le corpus « Articles », 2 923
  articles de Wikipédia en français répartis en six rubriques, découpés en
  2 339 articles d'entraînement et 584 de test (licence CC BY-SA 4.0 ; chaque
  article garde son URL et son numéro de révision) ;
- `queries.jsonl` : 112 requêtes de recherche et leur article pertinent ;
- `qa.jsonl`, `qa_hard.jsonl` : questions sur les articles (RAG) ;
- `qa_nav.jsonl`, `qa_rapports.jsonl` : questions sur les quatre rapports.

À télécharger :

- **Les quatre rapports publics** (chapitres 8, 36, 40, 41, 47) : voir
  `data/raw/rapports/README.md`, puis
  `uv run --group web python -m guide_nlp.corpus.rapports`.
- **frWac** (chapitres 17, 18, 20) : vecteurs word2vec de J.-P. Fauconnier,
  [frWac_non_lem_no_postag_no_phrase_500_skip_cut100.bin](https://embeddings.net/embeddings/frWac_non_lem_no_postag_no_phrase_500_skip_cut100.bin),
  à enregistrer sous `data/raw/pretrained/frwac.bin`.
- **Imagenette** (chapitres 42 et 43) :
  [imagenette2-320.tgz](https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz),
  à extraire dans `data/raw/imagenette/`.
- Les modèles de Hugging Face se téléchargent seuls au premier lancement.

Pour reconstruire le corpus à partir de Wikipédia (le résultat différera un
peu de celui du livre, car Wikipédia change chaque jour) :

```sh
uv run --group web python -m guide_nlp.corpus.collect
uv run --group web python scripts/build_corpus.py
```

## Reproduire les chiffres du livre

Les chiffres du livre ont été mesurés par les scripts d'expérience dont ce code
est tiré, sur une carte NVIDIA GB10, avec les versions du fichier `uv.lock`.
Relancés ici :

- tout ce qui se calcule sur CPU sans hasard (chapitres 3 à 15, comptages,
  BM25, TF-IDF, LSA, k-means, encodages de phrases) donne les mêmes valeurs,
  à la troisième décimale ;
- les entraînements sur GPU varient d'une exécution à l'autre de quelques
  millièmes (le calcul parallèle n'est pas déterministe) ;
- gensim, avec `workers=8`, n'est pas déterministe non plus : les analogies
  et les F1 des vecteurs entraînés sur notre corpus varient d'environ 0,01 ;
- les réponses d'Ollama dépendent un peu de sa version et de son réglage
  `OLLAMA_NUM_PARALLEL` (une réponse sur 55, par exemple, au chapitre 35).

## Qualité

```sh
uv run ruff check src scripts tests
uv run ruff format --check src scripts tests
uv run ty check
uv run pytest                                  # tests rapides
uv run --all-groups pytest src/guide_nlp/chapter46/test_behaviour.py
uv run --all-groups pytest src/guide_nlp/chapter41/test_rag.py   # Ollama requis
```

Le code utilise des `dataclass` pour les objets simples, `pydantic` pour tout
ce qui se valide (réglages, enregistrements des corpus, sorties structurées
des LLM) et des annotations de type partout.

## Licence

Code : MIT. Textes du corpus « Articles » : CC BY-SA 4.0 (Wikipédia).
