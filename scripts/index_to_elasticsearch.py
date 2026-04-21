"""
Bulk-index embeddings into Elasticsearch: document _source holds only `embedding`;
`_id` is `image_id` so kNN hits return the key you join on in PostgreSQL.

Query flow: encode query image -> ES kNN -> read `_id` from each hit -> SELECT * FROM images WHERE image_id = ...

Requires rows already in Postgres (e.g. scripts/embed_to_postgres.py). Vectors must be
L2-normalized (Dinov2Vectorizer default) for dot_product kNN to match cosine similarity.

Env:
  DATABASE_URL    — PostgreSQL connection string
  ELASTICSEARCH_URL — e.g. http://localhost:9200
  ES_INDEX_NAME   — optional, default images_knn
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
import psycopg
from pgvector.psycopg import register_vector
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")

MAPPING_PATH = _PROJECT_ROOT / "db" / "elasticsearch" / "images_knn_mapping.json"

SELECT_SQL = """
SELECT image_id, embedding
FROM images
WHERE embedding IS NOT NULL
  AND model_version IS NOT NULL
"""


def _ensure_index(es: Elasticsearch, index_name: str) -> None:
    if es.indices.exists(index=index_name):
        return
    with MAPPING_PATH.open(encoding="utf-8") as f:
        body = json.load(f)
    es.indices.create(
        index=index_name,
        settings=body["settings"],
        mappings=body["mappings"],
    )


def _iter_docs(
    index_name: str,
    rows: list[tuple[str, object]],
):
    for image_id, emb in rows:
        vec = emb.tolist() if hasattr(emb, "tolist") else list(emb)
        yield {
            "_op_type": "index",
            "_index": index_name,
            "_id": image_id,
            "_source": {"embedding": vec},
        }


def main() -> None:
    dsn = os.getenv("DATABASE_URL", "").strip()
    es_url = os.getenv("ELASTICSEARCH_URL", "").strip()
    index_name = os.getenv("ES_INDEX_NAME", "images_knn").strip() or "images_knn"

    if not dsn or not es_url:
        print(
            "Set DATABASE_URL and ELASTICSEARCH_URL in .env or the environment.",
            file=sys.stderr,
        )
        sys.exit(1)

    es = Elasticsearch(es_url)

    with psycopg.connect(dsn) as conn:
        register_vector(conn)
        with conn.cursor() as cur:
            cur.execute(SELECT_SQL)
            rows = list(cur.fetchall())

    if not rows:
        print("No rows with embeddings in PostgreSQL.", file=sys.stderr)
        return

    _ensure_index(es, index_name)
    ok, errors = bulk(es, _iter_docs(index_name, rows), raise_on_error=False)
    if errors:
        print(f"bulk(): indexed={ok}, errors={errors}", file=sys.stderr)
        sys.exit(1)

    indexed_ids = [r[0] for r in rows]
    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE images
                SET es_indexed_at = now(), es_index_name = %s
                WHERE image_id = ANY(%s)
                """,
                (index_name, indexed_ids),
            )
        conn.commit()

    print(
        f"Indexed {len(rows)} doc(s) into {index_name!r}; _id = image_id, "
        f"_source has only `embedding`.",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
