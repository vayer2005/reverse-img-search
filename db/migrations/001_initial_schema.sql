-- Phase B: PostgreSQL catalog for reverse image search (plan.md).
-- DINOv2 ViT-B/14 → embedding dimension 768 (facebook/dinov2-base).

CREATE EXTENSION IF NOT EXISTS vector;

-- Canonical image catalog: embeddings are source of truth; ES is rebuildable from here.
CREATE TABLE images (
    image_id TEXT PRIMARY KEY,
    -- Load source: inline bytes and/or durable URI (importer may set one or both).
    image_bytes BYTEA,
    storage_uri TEXT,
    mime_type TEXT,
    byte_length BIGINT CHECK (byte_length IS NULL OR byte_length >= 0),
    -- Corpus-specific fields (filename, labels, source dataset id, etc.)
    meta JSONB NOT NULL DEFAULT '{}'::jsonb,
    -- pgvector: D = 768 for DINOv2 ViT-B/14 CLS embedding
    embedding vector(768),
    model_version TEXT,
    embedded_at TIMESTAMPTZ,
    es_indexed_at TIMESTAMPTZ,
    es_index_name TEXT,
    needs_reindex BOOLEAN NOT NULL DEFAULT true,
    embed_error TEXT,
    quarantined_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT images_id_sha256_hex CHECK (
        image_id ~ '^[0-9a-f]{64}$'
    ),
    CONSTRAINT images_embedding_filled CHECK (
        embedding IS NULL
        OR (embedded_at IS NOT NULL AND model_version IS NOT NULL)
    )
);

COMMENT ON TABLE images IS 'Catalog of corpus images; embeddings are canonical; ES mirrors for query-time kNN.';
COMMENT ON COLUMN images.image_id IS 'Stable id: hex(SHA-256) over file bytes (64 lowercase hex chars).';
COMMENT ON COLUMN images.embedding IS 'DINOv2 ViT-B/14 CLS vector, dimension 768.';
COMMENT ON COLUMN images.needs_reindex IS 'True until embedding is written for current model_version policy.';
COMMENT ON COLUMN images.es_indexed_at IS 'Last successful bulk index to Elasticsearch for this row.';
COMMENT ON COLUMN images.es_index_name IS 'ES index name written on last successful index (alias cutover / upgrades).';

CREATE INDEX idx_images_needs_reindex
    ON images (needs_reindex, image_id)
    WHERE needs_reindex = true;

CREATE INDEX idx_images_quarantine
    ON images (quarantined_at)
    WHERE quarantined_at IS NOT NULL;

CREATE INDEX idx_images_meta_gin
    ON images USING gin (meta);

CREATE OR REPLACE FUNCTION images_set_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_images_set_updated_at
    BEFORE UPDATE ON images
    FOR EACH ROW
    EXECUTE PROCEDURE images_set_updated_at();
