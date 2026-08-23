import os
import logging
from typing import List, Optional
from google import genai
from google.genai import types
from google.genai.errors import APIError, ClientError

from backend.core.config import get_settings

logger = logging.getLogger("datapilot.rag")

EXPECTED_DIMENSION = 1536


class EmbeddingService:
    """
    RAG Embedding Service responsible for transforming text into dense vector embeddings
    using Google Gemini Embedding models via the official google-genai SDK.
    Matching Supabase pgvector column dimension (1536).
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gemini-embedding-001",
        fallback_model_name: str = "text-embedding-004",
        expected_dimension: int = EXPECTED_DIMENSION,
    ):
        settings = get_settings()
        self.api_key = api_key or settings.gemini_api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key or self.api_key.startswith("your_"):
            raise ValueError(
                "Gemini API Key is missing. Set GEMINI_API_KEY in your environment or .env file."
            )

        self.model_name = model_name
        self.fallback_model_name = fallback_model_name
        self.expected_dimension = expected_dimension
        self.client = genai.Client(api_key=self.api_key)

    def generate_embedding(self, text: str) -> List[float]:
        """
        Generates a 1536-dimensional embedding vector for the provided text payload.

        :param text: Non-empty retrieval text payload to embed.
        :return: List of 1536 float values representing the dense embedding vector.
        :raises ValueError: If input text is empty or returned vector dimension is incorrect.
        :raises Exception: If API call fails after retries.
        """
        if not text or not text.strip():
            raise ValueError("Cannot generate embedding for empty or whitespace-only text.")

        cfg = types.EmbedContentConfig(output_dimensionality=self.expected_dimension)

        try:
            res = self.client.models.embed_content(
                model=self.model_name,
                contents=text,
                config=cfg,
            )
            embedding_vals = self._extract_values(res)
        except (APIError, ClientError) as e:
            err_msg = str(e)
            if "404" in err_msg or "NOT_FOUND" in err_msg or "not found" in err_msg:
                logger.warning(
                    f"Model '{self.model_name}' returned 404; trying fallback model '{self.fallback_model_name}'"
                )
                res = self.client.models.embed_content(
                    model=self.fallback_model_name,
                    contents=text,
                    config=cfg,
                )
                embedding_vals = self._extract_values(res)
            else:
                logger.error(f"Gemini API error during embedding generation: {e}")
                raise
        except Exception as exc:
            logger.error(f"Unexpected error during embedding generation: {exc}")
            raise

        # Strict validation of vector dimension
        if len(embedding_vals) != self.expected_dimension:
            raise ValueError(
                f"Embedding dimension mismatch: expected {self.expected_dimension}, got {len(embedding_vals)}"
            )

        return [float(x) for x in embedding_vals]

    def _extract_values(self, res) -> List[float]:
        """Extracts float vector list from EmbedContentResponse object across SDK variants."""
        if hasattr(res, "embedding") and res.embedding and hasattr(res.embedding, "values"):
            return res.embedding.values
        if hasattr(res, "embeddings") and res.embeddings and len(res.embeddings) > 0:
            if hasattr(res.embeddings[0], "values"):
                return res.embeddings[0].values
        raise ValueError("Could not extract embedding values from Gemini API response.")
