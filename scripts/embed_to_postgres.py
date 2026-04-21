"""
Batch-embed image files and upsert into PostgreSQL (pgvector), matching db/migrations/001_initial_schema.sql.

Needs:
  - DATABASE_URL in .env (or environment), e.g. postgresql://user:pass@localhost:5432/dbname
  - IMG_PATH — same as vector_gen.py (directory of image files)

Later: bulk-index the same rows to Elasticsearch (dense_vector kNN) using embedding + image_id from PG.
"""

from __future__ import annotations

import mimetypes
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
import psycopg
from pgvector.psycopg import register_vector
from psycopg.types.json import Json

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")

# Run as: python scripts/embed_to_postgres.py  (so `scripts` is on sys.path for this import)
from vector_gen import (
    IMAGE_DIR,
    Dinov2Vectorizer,
    MODEL_ID,
    _iter_image_files,
    file_sha256_hex,
)


UPSERT_SQL = """
INSERT INTO images (
    image_id,
    embedding,
    model_version,
    embedded_at,
    meta,
    storage_uri,
    byte_length,
    mime_type
) VALUES (
    %s,
    %s,
    %s,
    now(),
    %s,
    %s,
    %s,
    %s
)
ON CONFLICT (image_id) DO UPDATE SET
    embedding = EXCLUDED.embedding,
    model_version = EXCLUDED.model_version,
    embedded_at = EXCLUDED.embedded_at,
    meta = EXCLUDED.meta,
    storage_uri = COALESCE(EXCLUDED.storage_uri, images.storage_uri),
    byte_length = EXCLUDED.byte_length,
    mime_type = COALESCE(EXCLUDED.mime_type, images.mime_type);
"""


def main() -> None:
    dsn = os.getenv("DATABASE_URL", "").strip()
    if not dsn:
        print(
            "Set DATABASE_URL in .env or the environment, e.g.\n"
            "  DATABASE_URL=postgresql://user:pass@localhost:5432/yourdb",
            file=sys.stderr,
        )
        sys.exit(1)

    if IMAGE_DIR is None or not IMAGE_DIR.is_dir():
        print(
            "Set IMG_PATH in .env to an existing directory of images (same as vector_gen.py).",
            file=sys.stderr,
        )
        sys.exit(1)

    paths = _iter_image_files(IMAGE_DIR)
    if not paths:
        print(f"No files in {IMAGE_DIR}", file=sys.stderr)
        return

    vec = Dinov2Vectorizer()
    model_version = MODEL_ID

    with psycopg.connect(dsn) as conn:
        register_vector(conn)
        for path in paths:
            image_id = file_sha256_hex(path)
            raw = path.read_bytes()
            emb = vec.encode_one(path)
            meta = {"filename": path.name, "stem": path.stem}
            storage_uri = path.resolve().as_uri()
            byte_length = len(raw)
            mime_type = mimetypes.guess_type(path.name)[0]
            with conn.cursor() as cur:
                cur.execute(
                    UPSERT_SQL,
                    (
                        image_id,
                        emb,
                        model_version,
                        Json(meta),
                        storage_uri,
                        byte_length,
                        mime_type,
                    ),
                )
            conn.commit()
            print(f"OK {path.name} -> {image_id[:12]}…", file=sys.stderr)

    print(f"Indexed {len(paths)} image(s) into PostgreSQL.", file=sys.stderr)


if __name__ == "__main__":
    main()
