-- Migration: Create pgvector similarity search function for Evindra RAG system
-- Description: Performs cosine similarity search against rag_documents.embedding (1536-dim vector)

DROP FUNCTION IF EXISTS public.match_rag_documents(vector, int);
DROP FUNCTION IF EXISTS public.match_rag_documents(float8[], int);

CREATE OR REPLACE FUNCTION public.match_rag_documents (
  query_embedding float8[],
  match_count int DEFAULT 5
)
RETURNS TABLE (
  rag_id text,
  scenario_id text,
  domain text,
  scenario_type text,
  retrieval_text text,
  metadata jsonb,
  similarity float
)
LANGUAGE plpgsql
AS $$
DECLARE
  query_vec vector(1536) := query_embedding::vector(1536);
BEGIN
  RETURN QUERY
  SELECT
    r.rag_id::text,
    r.scenario_id::text,
    r.domain::text,
    r.scenario_type::text,
    r.retrieval_text::text,
    r.metadata::jsonb,
    (1 - (r.embedding <=> query_vec))::float AS similarity
  FROM rag_documents r
  WHERE r.embedding IS NOT NULL
  ORDER BY r.embedding <=> query_vec
  LIMIT match_count;
END;
$$;

GRANT EXECUTE ON FUNCTION public.match_rag_documents(float8[], int) TO anon, authenticated, service_role;
NOTIFY pgrst, 'reload schema';
