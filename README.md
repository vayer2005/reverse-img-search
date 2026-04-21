# reverse-img-search

Corpus images are embedded with **DINOv2** (`facebook/dinov2-base`, 768-d CLS, L2-normalized), stored in **PostgreSQL + pgvector**, and indexed in **Elasticsearch** for kNN (`dense_vector`, dot product). Search returns `image_id` from Elasticsearch; metadata and paths come from Postgres.

**Scripts (from repo root):**

- `python3 scripts/embed_to_postgres.py`: embed files under `IMG_PATH` into Postgres.
- `python3 scripts/index_to_elasticsearch.py`: bulk-index embeddings into `ES_INDEX_NAME` (default `images_knn`).
- `python3 scripts/find_similar.py <query.jpg> [--top-k N]`: encode the query and run kNN.

Configure `.env`: `IMG_PATH`, `DATABASE_URL`, `ELASTICSEARCH_URL`, optional `ES_INDEX_NAME`.

---

## Lighthouse query experiment

Images are stored under [`run_exampes/`](run_exampes/) so paths work on GitHub and locally.

**Query** (not in the corpus):

<p align="center">
  <img src="run_exampes/lighthouse.jpg" alt="Query image: lighthouse at the shore" width="520">
</p>

**Top-1 match** from corpus (`20077.jpg`):

<p align="center">
  <img src="run_exampes/lighthouse_top1_corpus_20077.jpg" alt="Most similar corpus image 20077.jpg" width="520">
</p>

- **Command:** `python3 scripts/find_similar.py run_exampes/lighthouse.jpg --top-k 5`
- **Top-1 result:** corpus file `20077.jpg`, dot-product score **≈ 0.822** (`image_id` `b5f04aece150f3264667de7699cb44ec38d62c8daf6198a3240d7bcf970a451f`).
- The index contained **3000** images under `IMG_PATH` when this was run; scores depend on corpus contents and model.

---

## Random building query experiment

**Query** (not in the corpus):

<p align="center">
  <img src="run_exampes/random_building.jpg" alt="Query image: random building" width="520">
</p>

**Top-1 match** from corpus (`20250.jpg`):

<p align="center">
  <img src="run_exampes/random_building_top1_corpus_20250.jpg" alt="Most similar corpus image 20250.jpg" width="520">
</p>

- **Command:** `python3 scripts/find_similar.py run_exampes/random_building.jpg --top-k 5`
- **Top-1 result:** corpus file `20250.jpg`, dot-product score **≈ 0.663** (`image_id` `619ddd33ec10fa9bd6ee1dbfd6ac804008b49940cd398aecb2971ad8a7b2d642`).

**Top 5** (corpus filename and dot-product score):

| Rank | File | Score |
| ---: | --- | ---: |
| 1 | `20250.jpg` | 0.663 |
| 2 | `20436.jpg` | 0.632 |
| 3 | `21948.jpg` | 0.620 |
| 4 | `23660.jpg` | 0.619 |
| 5 | `22801.jpg` | 0.601 |

Same **3000**-image index as the lighthouse run; scores are only comparable within the same model and corpus.
