"""
Given a path to an image file, encode with DINOv2 and run Elasticsearch kNN on the corpus index.

Uses the same embedding path as indexing (Dinov2Vectorizer + dot_product on L2-normalized vectors).
By default excludes the query image from results when it is already indexed (same SHA-256 `image_id`).

Env:
  ELASTICSEARCH_URL — required, e.g. http://localhost:9200
  ES_INDEX_NAME     — optional, default images_knn
  DATABASE_URL      — optional; if set, enriches hits with storage_uri + meta from PostgreSQL

Run:
  python scripts/find_similar.py /path/to/query.jpg
  python scripts/find_similar.py /path/to/query.jpg --top-k 5
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
import psycopg
from elasticsearch import Elasticsearch

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")

from vector_gen import Dinov2Vectorizer, file_sha256_hex


PG_ENRICH_SQL = """
SELECT image_id, storage_uri, meta
FROM images
WHERE image_id = ANY(%s);
"""


def _enrich_hits(dsn: str, image_ids: list[str]) -> dict[str, dict[str, Any]]:
    if not image_ids:
        return {}
    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(PG_ENRICH_SQL, (image_ids,))
            rows = cur.fetchall()
    out: dict[str, dict[str, Any]] = {}
    for image_id, storage_uri, meta in rows:
        out[image_id] = {"storage_uri": storage_uri, "meta": meta}
    return out


def similar_images(
    image_path: str | Path,
    *,
    es: Elasticsearch | None = None,
    index_name: str | None = None,
    es_url: str | None = None,
    vectorizer: Dinov2Vectorizer | None = None,
    top_k: int = 1,
    exclude_self: bool = True,
    enrich_from_postgres: bool = True,
) -> list[dict[str, Any]]:
    """
    Return ranked hits: `image_id` (from ES `_id`), `score`, optionally `storage_uri`, `meta` from Postgres.
    """
    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(path)

    url = es_url or os.getenv("ELASTICSEARCH_URL", "").strip()
    if es is None and not url:
        raise ValueError("ELASTICSEARCH_URL is not set and no Elasticsearch client was passed.")
    client = es or Elasticsearch(url)
    idx = index_name or os.getenv("ES_INDEX_NAME", "images_knn").strip() or "images_knn"

    v = vectorizer or Dinov2Vectorizer()
    qvec = v.encode_one(path).tolist()

    query_id: str | None = None
    if exclude_self:
        query_id = file_sha256_hex(path)

    knn: dict[str, Any] = {
        "field": "embedding",
        "query_vector": qvec,
        "k": top_k,
        "num_candidates": max(100, top_k * 20),
    }
    if query_id:
        knn["filter"] = {
            "bool": {
                "must_not": [{"ids": {"values": [query_id]}}],
            }
        }

    resp = client.search(
        index=idx,
        knn=knn,
        size=top_k,
        source=False,
    )

    hits = resp.get("hits", {}).get("hits", [])
    results: list[dict[str, Any]] = []
    for h in hits:
        results.append(
            {
                "image_id": h["_id"],
                "score": float(h.get("_score", 0.0)),
            }
        )

    dsn = os.getenv("DATABASE_URL", "").strip()
    if enrich_from_postgres and dsn and results:
        ids = [r["image_id"] for r in results]
        extra = _enrich_hits(dsn, ids)
        for r in results:
            e = extra.get(r["image_id"])
            if e:
                r["storage_uri"] = e["storage_uri"]
                r["meta"] = e["meta"]

    return results


def main() -> None:
    p = argparse.ArgumentParser(description="Find most similar indexed images for a query file.")
    p.add_argument("image_path", type=Path, help="Path to query image (JPEG/PNG, etc.)")
    p.add_argument(
        "--top-k",
        type=int,
        default=1,
        help="Number of nearest neighbors to return (default: 1).",
    )
    p.add_argument(
        "--include-self",
        action="store_true",
        help="If the query file is indexed, still allow it as a hit (same image_id).",
    )
    p.add_argument(
        "--no-pg",
        action="store_true",
        help="Do not join PostgreSQL for storage_uri / meta.",
    )
    args = p.parse_args()

    try:
        hits = similar_images(
            args.image_path,
            top_k=max(1, args.top_k),
            exclude_self=not args.include_self,
            enrich_from_postgres=not args.no_pg,
        )
    except FileNotFoundError as e:
        print(e, file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(e, file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Search failed: {e}", file=sys.stderr)
        sys.exit(1)

    print(json.dumps(hits, indent=2))


if __name__ == "__main__":
    main()
