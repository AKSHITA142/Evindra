import os
import sys
import logging
from typing import List, Dict, Any, Optional

from backend.core.config import get_settings

# Cloud-only dependencies — guarded so the module is importable without them.
try:
    import dotenv
except ImportError:
    dotenv = None  # type: ignore[assignment]

try:
    import psycopg2
    _PSYCOPG2_AVAILABLE = True
except ImportError:
    psycopg2 = None  # type: ignore[assignment]
    _PSYCOPG2_AVAILABLE = False

try:
    from supabase import create_client, Client
    _SUPABASE_AVAILABLE = True
except ImportError:
    create_client = None  # type: ignore[assignment]
    Client = None  # type: ignore[assignment,misc]
    _SUPABASE_AVAILABLE = False

try:
    from backend.services.rag.embedding_service import EmbeddingService, EXPECTED_DIMENSION
    _EMBEDDING_AVAILABLE = True
except ImportError:
    EmbeddingService = None  # type: ignore[assignment,misc]
    EXPECTED_DIMENSION = 1536
    _EMBEDDING_AVAILABLE = False

logger = logging.getLogger("datapilot.rag.retrieval")

# Fallback env path for Evindra cloud database configuration
EVINDRA_ENV_PATH = "/Users/akshitajariwala/Desktop/Prime_Classes/Evindra Testing/.env"

# Direct PostgreSQL fallback credentials for Supabase pgvector queries
PG_HOST = "aws-0-ap-northeast-1.pooler.supabase.com"
PG_PORT = 5432
PG_USER = "postgres.muvbzqqxrgthbdgutaap"
PG_PASS = "MananPatel@7310"
PG_DB = "postgres"


def get_supabase_client() -> Client:
    """
    Initializes and returns Supabase client targeting the active Evindra project.
    Checks environment variables first, falling back to local .env or Evindra environment file.
    """
    url = os.getenv("EVINDRA_SUPABASE_URL") or os.getenv("SUPABASE_URL")
    key = os.getenv("EVINDRA_SUPABASE_KEY") or os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_KEY")

    if not url or not key:
        settings = get_settings()
        url = url or settings.supabase_url
        key = key or settings.supabase_key

    if (not url or not key) and os.path.exists(EVINDRA_ENV_PATH):
        cfg = dotenv.dotenv_values(EVINDRA_ENV_PATH)
        url = url or cfg.get("SUPABASE_URL")
        key = key or cfg.get("SUPABASE_SERVICE_ROLE_KEY") or cfg.get("SUPABASE_KEY")

    if not url or not key:
        raise RuntimeError(
            "Could not load Supabase URL and Key for Vector Retrieval. "
            "Ensure SUPABASE_URL and SUPABASE_KEY or EVINDRA_ENV_PATH are set."
        )

    return create_client(url, key)


class VectorRetrievalService:
    """
    Independent Vector Retrieval Service for the Evindra RAG system.
    Transforms raw ML query text into 1536-dimensional vector embeddings via Gemini EmbeddingService,
    and executes pgvector cosine similarity search against rag_documents in Supabase.
    
    Treats existing scenario embeddings as READ-ONLY.
    """

    def __init__(
        self,
        embedding_service: Optional[EmbeddingService] = None,
        supabase_client: Optional[Client] = None,
    ):
        self.embedding_service = embedding_service or EmbeddingService()
        self.sp = supabase_client or get_supabase_client()

    def search_similar_scenarios(
        self,
        query_text: str,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Retrieves the Top-K most similar previously generated scenarios for a given ML query.

        Args:
            query_text: Natural language description of an ML/data-preprocessing problem.
            top_k: Number of top matching scenarios to return (default: 5).

        Returns:
            List of dictionaries containing:
            - scenario_id
            - domain
            - scenario_type
            - retrieval_text
            - metadata
            - similarity_score (float between 0.0 and 1.0)
        """
        if not query_text or not query_text.strip():
            raise ValueError("Query text cannot be empty for vector retrieval.")

        if top_k <= 0:
            raise ValueError(f"top_k must be a positive integer, got: {top_k}")

        logger.info(f"Generating query embedding for retrieval query (length={len(query_text)})...")

        # Step 1: Generate query embedding using existing EmbeddingService
        try:
            query_vector = self.embedding_service.generate_embedding(query_text)
        except Exception as e:
            logger.error(f"Failed to generate query embedding: {e}")
            raise RuntimeError(f"Query embedding generation failed: {e}") from e

        # Step 2: Validate embedding dimension strictly (must be 1536)
        if len(query_vector) != EXPECTED_DIMENSION:
            raise ValueError(
                f"Generated query vector dimension mismatch: expected {EXPECTED_DIMENSION}, got {len(query_vector)}"
            )

        logger.info(
            f"Query embedding generated successfully (dim={len(query_vector)}). Executing pgvector search (top_k={top_k})..."
        )

        # Step 3: Execute pgvector cosine similarity search via Supabase RPC with Postgres fallback
        raw_results = self._execute_vector_search(query_vector, top_k)
        logger.info(f"Vector search returned {len(raw_results)} candidate scenarios.")

        # Step 4: Parse and return clean scenario objects
        formatted_results: List[Dict[str, Any]] = []
        for row in raw_results:
            sim_raw = row.get("similarity")
            sim_score = 0.0
            if sim_raw is not None:
                try:
                    sim_score = float(sim_raw)
                    if sim_score != sim_score:  # Handle NaN gracefully
                        sim_score = 0.0
                except (ValueError, TypeError):
                    sim_score = 0.0

            formatted_results.append({
                "scenario_id": row.get("scenario_id"),
                "domain": row.get("domain"),
                "scenario_type": row.get("scenario_type"),
                "retrieval_text": row.get("retrieval_text"),
                "metadata": row.get("metadata", {}),
                "similarity_score": round(sim_score, 6),
            })

        # Ensure results are sorted descending by similarity_score
        formatted_results.sort(key=lambda x: x["similarity_score"], reverse=True)
        return formatted_results

    def _execute_vector_search(self, query_vector: List[float], top_k: int) -> List[Dict[str, Any]]:
        """Executes vector search via Supabase RPC first, with fallback to direct Postgres pgvector query."""
        # Try Supabase RPC method first
        try:
            rpc_res = self.sp.rpc(
                "match_rag_documents",
                {
                    "query_embedding": query_vector,
                    "match_count": top_k,
                },
            ).execute()
            if rpc_res.data is not None:
                return rpc_res.data
        except Exception as rpc_err:
            logger.warning(f"Supabase RPC search hit exception: {rpc_err}. Falling back to direct PostgreSQL query...")

        # Fallback: Direct PostgreSQL pgvector query via psycopg2
        try:
            conn = psycopg2.connect(
                host=PG_HOST,
                port=PG_PORT,
                user=PG_USER,
                password=PG_PASS,
                dbname=PG_DB,
                sslmode="require",
                connect_timeout=10,
            )
            cur = conn.cursor()
            vec_str = "[" + ",".join(str(f) for f in query_vector) + "]"

            sql = """
                SELECT
                    r.rag_id::text,
                    r.scenario_id::text,
                    r.domain::text,
                    r.scenario_type::text,
                    r.retrieval_text::text,
                    r.metadata::jsonb,
                    (1 - (r.embedding <=> %s::vector(1536)))::float AS similarity
                FROM rag_documents r
                WHERE r.embedding IS NOT NULL
                ORDER BY r.embedding <=> %s::vector(1536)
                LIMIT %s;
            """
            cur.execute(sql, (vec_str, vec_str, top_k))
            rows = cur.fetchall()

            results = []
            for row in rows:
                results.append({
                    "rag_id": row[0],
                    "scenario_id": row[1],
                    "domain": row[2],
                    "scenario_type": row[3],
                    "retrieval_text": row[4],
                    "metadata": row[5],
                    "similarity": row[6],
                })
            conn.close()
            return results
        except Exception as pg_err:
            logger.error(f"Direct PostgreSQL vector search failed: {pg_err}")
            raise RuntimeError(f"Vector search failed on both Supabase RPC and direct PostgreSQL: {pg_err}") from pg_err


# Convenience function for direct module usage
def search_similar_scenarios(query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """Convenience wrapper around VectorRetrievalService.search_similar_scenarios."""
    service = VectorRetrievalService()
    return service.search_similar_scenarios(query_text, top_k=top_k)
