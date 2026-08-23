import os
import re
import math
import time
import logging
from typing import List, Optional
from google import genai
from google.genai import types
from google.genai.errors import APIError, ClientError

from backend.core.config import get_settings

logger = logging.getLogger("datapilot.rag")

PRIMARY_EMBEDDING_MODEL = "gemini-embedding-2"
EXPECTED_DIMENSION = 1536


class EmbeddingService:
    """
    RAG Embedding Service responsible for transforming text into dense vector embeddings
    using Google Gemini Embedding model (gemini-embedding-2) via the official google-genai SDK.
    Strictly enforced to match Supabase pgvector column dimension (1536).
    Cross-model fallback is disabled to prevent vector space contamination.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = PRIMARY_EMBEDDING_MODEL,
        expected_dimension: int = EXPECTED_DIMENSION,
    ):
        settings = get_settings()
        self.api_key = api_key or settings.gemini_api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key or self.api_key.startswith("your_"):
            raise ValueError(
                "Gemini API Key is missing. Set GEMINI_API_KEY in your environment or .env file."
            )

        self.model_name = model_name
        self.expected_dimension = expected_dimension
        self.client = genai.Client(api_key=self.api_key)

    def generate_embedding(self, text: str) -> List[float]:
        """
        Generates a 1536-dimensional embedding vector for a single text payload.
        """
        results = self.generate_embeddings_batch([text])
        return results[0]

    def generate_embeddings_batch(self, texts: List[str], max_retries: int = 5) -> List[List[float]]:
        """
        Generates 1536-dimensional embedding vectors for a batch of text payloads using gemini-embedding-2.
        Retries on transient rate limits (HTTP 429) using exponential backoff without switching models.
        """
        if not texts:
            return []

        cleaned_texts = [t if (t and t.strip()) else "empty" for t in texts]
        cfg = types.EmbedContentConfig(output_dimensionality=self.expected_dimension)

        last_exception = None
        for attempt in range(1, max_retries + 1):
            try:
                res = self.client.models.embed_content(
                    model=self.model_name,
                    contents=cleaned_texts if len(cleaned_texts) > 1 else cleaned_texts[0],
                    config=cfg,
                )
                return self._extract_batch_values(res, expected_count=len(cleaned_texts))
            except (APIError, ClientError) as e:
                last_exception = e
                err_msg = str(e)
                if "429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg or "Quota exceeded" in err_msg:
                    match = re.search(r"Please retry in (\d+(?:\.\d+)?)s", err_msg)
                    parsed_sec = math.ceil(float(match.group(1))) + 2 if match else 5 * (2 ** (attempt - 1))
                    logger.warning(
                        f"Embedding Model '{self.model_name}' Rate Limit (429) hit (attempt {attempt}/{max_retries}). "
                        f"Retrying in {parsed_sec}s on same model..."
                    )
                    if attempt < max_retries:
                        time.sleep(min(parsed_sec, 15))
                        continue
                else:
                    logger.error(f"Gemini API error during embedding generation with '{self.model_name}': {e}")
                    raise RuntimeError(f"Embedding generation with model '{self.model_name}' failed: {e}") from e
            except Exception as exc:
                last_exception = exc
                logger.error(f"Unexpected error during embedding generation with '{self.model_name}': {exc}")
                raise RuntimeError(f"Unexpected error during embedding generation with '{self.model_name}': {exc}") from exc

        raise RuntimeError(
            f"Embedding generation with configured model '{self.model_name}' failed after {max_retries} retries: {last_exception}"
        )


    def _extract_batch_values(self, res, expected_count: int = 1) -> List[List[float]]:
        """Extracts list of vector float lists from EmbedContentResponse object."""
        if hasattr(res, "embeddings") and res.embeddings:
            vectors = []
            for emb in res.embeddings:
                vals = [float(x) for x in emb.values]
                if len(vals) != self.expected_dimension:
                    raise ValueError(
                        f"Embedding dimension mismatch: expected {self.expected_dimension}, got {len(vals)}"
                    )
                vectors.append(vals)
            return vectors
        elif hasattr(res, "embedding") and res.embedding and hasattr(res.embedding, "values"):
            vals = [float(x) for x in res.embedding.values]
            if len(vals) != self.expected_dimension:
                raise ValueError(
                    f"Embedding dimension mismatch: expected {self.expected_dimension}, got {len(vals)}"
                )
            return [vals]

        raise ValueError("Could not extract embedding values from Gemini API response.")
