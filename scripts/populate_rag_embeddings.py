#!/usr/bin/env python3
"""
Populate RAG Embeddings Script for Evindra

Reads rag_documents records from Supabase where embedding IS NULL,
generates 768-dimensional dense vector embeddings via EmbeddingService (Gemini text-embedding-004 / gemini-embedding-001),
and updates ONLY the embedding column for each record in Supabase.

Usage:
  python scripts/populate_rag_embeddings.py --limit 10       # Dry-run mode (10 records)
  python scripts/populate_rag_embeddings.py --batch-size 50   # Normal run
"""

import sys
import os
import argparse
import time
import logging
from typing import List, Dict, Any, Optional

import dotenv
from supabase import create_client, Client

# Ensure workspace root is in sys.path
WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from backend.services.rag.embedding_service import EmbeddingService, EXPECTED_DIMENSION

# Setup logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("populate_rag_embeddings")

# Path to Evindra Testing env (containing https://muvbzqqxrgthbdgutaap.supabase.co credentials)
EVINDRA_ENV_PATH = "/Users/akshitajariwala/Desktop/Prime_Classes/Evindra Testing/.env"


def get_supabase_client() -> Client:
    """Initializes and returns Supabase client targeting the Evindra project."""
    url = os.getenv("EVINDRA_SUPABASE_URL") or os.getenv("SUPABASE_URL")
    key = os.getenv("EVINDRA_SUPABASE_KEY") or os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_KEY")

    if (not url or not key) and os.path.exists(EVINDRA_ENV_PATH):
        cfg = dotenv.dotenv_values(EVINDRA_ENV_PATH)
        url = url or cfg.get("SUPABASE_URL")
        key = key or cfg.get("SUPABASE_SERVICE_ROLE_KEY") or cfg.get("SUPABASE_KEY")

    if not url or not key:
        raise RuntimeError(
            "Could not load Supabase URL and Key. Ensure EVINDRA_ENV_PATH or environment variables are set."
        )

    logger.info(f"Connecting to Supabase project: {url}")
    return create_client(url, key)


def fetch_unembedded_records(
    sp: Client,
    limit: Optional[int] = None,
    force_all: bool = False
) -> List[Dict[str, Any]]:
    """
    Fetches records from rag_documents where embedding IS NULL (or all if force_all=True).
    """
    query = sp.table("rag_documents").select("rag_id, scenario_id, domain, scenario_type, retrieval_text, embedding")

    if not force_all:
        query = query.is_("embedding", "null")

    # Order deterministically by created_at / rag_id
    query = query.order("rag_id")

    if limit and limit > 0:
        query = query.limit(limit)

    res = query.execute()
    return res.data or []


def get_embedding_counts(sp: Client) -> Dict[str, int]:
    """Returns total, NULL embedding, and non-NULL embedding counts from rag_documents."""
    total_res = sp.table("rag_documents").select("rag_id", count="exact").execute()
    null_res = sp.table("rag_documents").select("rag_id", count="exact").is_("embedding", "null").execute()
    non_null_res = sp.table("rag_documents").select("rag_id", count="exact").not_.is_("embedding", "null").execute()

    return {
        "total": total_res.count or 0,
        "null_count": null_res.count or 0,
        "non_null_count": non_null_res.count or 0,
    }


def embed_and_update_record(
    sp: Client,
    embedding_service: EmbeddingService,
    record: Dict[str, Any],
    max_retries: int = 3
) -> bool:
    """
    Generates embedding for a single record and updates ONLY the embedding column in Supabase.
    Retries up to max_retries on transient failures.
    """
    rag_id = record["rag_id"]
    scenario_id = record.get("scenario_id", "unknown")
    text = record.get("retrieval_text")

    if not text or not text.strip():
        logger.error(f"Record rag_id={rag_id} (scenario={scenario_id}) has empty retrieval_text; skipping.")
        return False

    for attempt in range(1, max_retries + 1):
        try:
            vec = embedding_service.generate_embedding(text)
            
            # Validate vector dimension strictly before DB update
            if len(vec) != EXPECTED_DIMENSION:
                raise ValueError(f"Vector dimension {len(vec)} != expected {EXPECTED_DIMENSION}")

            # Update ONLY the embedding column in Supabase
            sp.table("rag_documents").update({"embedding": vec}).eq("rag_id", rag_id).execute()
            return True
        except Exception as exc:
            logger.warning(
                f"Attempt {attempt}/{max_retries} failed for rag_id={rag_id} ({scenario_id}): {exc}"
            )
            if attempt < max_retries:
                time.sleep(2 * attempt)  # Exponential backoff delay
            else:
                logger.error(f"FAILED permanently for rag_id={rag_id} after {max_retries} attempts.")
                return False

    return False


def main():
    parser = argparse.ArgumentParser(description="Populate RAG embeddings in Supabase rag_documents table.")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of records to process (e.g. --limit 10 for dry run)."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Batch size for progress reporting (default: 50)."
    )
    parser.add_argument(
        "--force-all",
        action="store_true",
        help="Process all records even if embedding is already populated (default: False)."
    )

    args = parser.parse_args()

    logger.info("==================================================")
    logger.info("  EVINDRA RAG EMBEDDING GENERATION PIPELINE")
    logger.info("==================================================")
    if args.limit:
        logger.info(f"⚡ RUNNING IN LIMITED MODE: Processing max {args.limit} records")

    # 1. Initialize Supabase & EmbeddingService
    sp = get_supabase_client()
    embedding_svc = EmbeddingService()

    # 2. Check baseline DB state
    baseline = get_embedding_counts(sp)
    logger.info(f"Baseline Database State:")
    logger.info(f"  - Total rag_documents: {baseline['total']}")
    logger.info(f"  - Currently Non-NULL embeddings: {baseline['non_null_count']}")
    logger.info(f"  - Currently NULL embeddings: {baseline['null_count']}")

    # 3. Fetch target records
    records = fetch_unembedded_records(sp, limit=args.limit, force_all=args.force_all)
    total_to_process = len(records)

    if total_to_process == 0:
        logger.info("🎉 No records requiring embedding generation found!")
        return

    logger.info(f"\nStarting embedding generation for {total_to_process} records...")
    
    success_count = 0
    failure_count = 0
    processed_details = []

    start_time = time.time()

    for idx, rec in enumerate(records, 1):
        rag_id = rec["rag_id"]
        scenario_id = rec.get("scenario_id", "unknown")

        ok = embed_and_update_record(sp, embedding_svc, rec)

        if ok:
            success_count += 1
            processed_details.append({
                "rag_id": rag_id,
                "scenario_id": scenario_id,
                "status": "SUCCESS",
                "dim": EXPECTED_DIMENSION,
            })
        else:
            failure_count += 1
            processed_details.append({
                "rag_id": rag_id,
                "scenario_id": scenario_id,
                "status": "FAILED",
                "dim": 0,
            })

        # Progress reporting per batch or item
        if idx % args.batch_size == 0 or idx == total_to_process:
            elapsed = time.time() - start_time
            rate = idx / elapsed if elapsed > 0 else 0
            logger.info(
                f"Progress: [{idx}/{total_to_process}] ({idx/total_to_process*100:.1f}%) | "
                f"Success: {success_count} | Failed: {failure_count} | Rate: {rate:.2f} rec/sec"
            )

    # 4. Final Post-run Verification & Counts
    after_run = get_embedding_counts(sp)
    elapsed_total = time.time() - start_time

    logger.info("\n==================================================")
    logger.info("  BATCH EMBEDDING GENERATION SUMMARY REPORT")
    logger.info("==================================================")
    logger.info(f"Execution Time: {elapsed_total:.2f} seconds")
    logger.info(f"Records Processed: {total_to_process}")
    logger.info(f"Successful Embeddings: {success_count}")
    logger.info(f"Failed Embeddings: {failure_count}")
    logger.info(f"\nUpdated Database State:")
    logger.info(f"  - Total rag_documents: {after_run['total']}")
    logger.info(f"  - Non-NULL Embeddings: {after_run['non_null_count']}")
    logger.info(f"  - Remaining NULL Embeddings: {after_run['null_count']}")

    if args.limit and args.limit <= 20:
        logger.info("\nProcessed Record Details:")
        for item in processed_details:
            logger.info(
                f"  - rag_id: {item['rag_id']} | scenario_id: {item['scenario_id']} | "
                f"status: {item['status']} | dimension: {item['dim']}"
            )


if __name__ == "__main__":
    main()
