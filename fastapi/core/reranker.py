# fastapi/core/reranker.py
#
# Cross-encoder reranker for the Phase 9 RAG pipeline.
#
# Two responsibilities:
#   1. load_reranker(model_name) — called once at startup via lifespan,
#      loads the cross-encoder model into the module-level singleton.
#   2. rerank(query, chunks, top_k) — scores each (query, chunk_text)
#      pair using joint cross-encoder attention and returns the top_k
#      highest-scoring chunks.
#
# This module is stateful: it holds a ~90MB model in memory.
# Keeping it isolated here means lifespan in main.py loads it once
# and chat.py calls rerank() without knowing anything about the model.

from sentence_transformers import CrossEncoder
import logging
from typing import Any

logger = logging.getLogger(__name__)
# __name__ resolves to "core.reranker" inside the container.
# All log lines from this module are prefixed with that path in
# docker compose logs, making them easy to grep.

_reranker: CrossEncoder | None = None
# Module-level singleton. None until load_reranker() is called.
# Underscore prefix signals private — callers use load_reranker()
# and rerank(), never this variable directly.


def load_reranker(model_name: str) -> None:
    """
    Load the cross-encoder model into the module-level singleton.

    Called exactly once during FastAPI lifespan startup, before the
    application accepts any requests. Loading here guarantees every
    subsequent call to rerank() finds the model already in memory.

    Why a parameter and not reading config directly?
    Testability — a test can call load_reranker("some-model") without
    mocking environment variables. The caller (main.py lifespan) owns
    the responsibility of reading the config value and passing it in.

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier.
        Expected: "cross-encoder/ms-marco-MiniLM-L-6-v2"
        On first call this downloads ~90MB from HuggingFace Hub and
        caches it in the container filesystem. Subsequent container
        restarts load from cache — no re-download.
    """
    global _reranker
    # global declaration is necessary because we are assigning to a
    # module-level variable inside a function. Without it, Python
    # creates a new local variable and the assignment is invisible
    # outside this function. This is one of the few legitimate uses
    # of global — a module-level singleton loaded once.

    logger.info(f"Loading reranker model: {model_name}")
    # Visible in docker compose logs fastapi_a before the model loads.
    # If the model name is wrong (typo in config), it appears here.

    _reranker = CrossEncoder(model_name, max_length=512)
    # CrossEncoder is the sentence-transformers class for cross-encoder
    # models. It is distinct from SentenceTransformer (bi-encoder).
    #   SentenceTransformer.encode() → one vector per text (independent)
    #   CrossEncoder.predict()       → one score per (query, text) pair
    #                                  (joint attention, more accurate)
    #
    # max_length=512: maximum token length per input pair
    # (query + [SEP] + chunk_text tokenized together as one sequence).
    # 512 is the BERT-family standard and matches this model's training
    # configuration. Chunks exceeding this are truncated at the tail.
    # Being explicit here is defensive — do not rely on the model default.

    logger.info("Reranker model loaded successfully")
    # Second log line confirms no exception was raised during loading.
    # The gap between the two log lines shows how long loading took.


def rerank(
    query: str,
    chunks: list[dict[str, Any]],
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """
    Score each chunk against the query and return the top_k most relevant.

    Called inside the LLM orchestration loop in chat.py, only when
    the tool called was query_knowledge_base. The other two tools
    (search_documents, get_document_summary) return structured metadata
    where relevance scoring makes no sense — reranking is skipped for them.

    Parameters
    ----------
    query : str
        The user's question exactly as received over the WebSocket.
        The same string that was used to retrieve the 20 Qdrant candidates.
    chunks : list[dict[str, Any]]
        The 20 candidate chunks as parsed from the query_knowledge_base
        tool result JSON. Each dict must contain at minimum:
          "text"           — the chunk content (fed to the cross-encoder)
          "original_filename", "section_label", "location_index"
                           — used for citations in chat.py
    top_k : int
        Number of chunks to return after reranking. Default 5.
        The caller passes settings.top_k_rerank, but the default here
        covers unit tests that call rerank() directly.

    Returns
    -------
    list[dict[str, Any]]
        The top_k chunk dicts sorted by reranker score descending.
        First element is the most relevant chunk.
        Scores are not included in the return value — the caller needs
        the chunk dicts for citation formatting, not the raw scores.
    """
    if _reranker is None:
        raise RuntimeError(
            "Reranker not loaded. Call load_reranker() at startup."
        )
    # Explicit guard. Without this, the next line crashes with:
    # AttributeError: 'NoneType' object has no attribute 'predict'
    # — harder to diagnose than this message.

    if not chunks:
        return []
    # Empty input: Qdrant returned zero results (empty corpus or all
    # chunks filtered below similarity threshold). Return immediately —
    # no model call needed, no index-out-of-bounds risk below.

    pairs = [(query, chunk["text"]) for chunk in chunks]
    # Build the input pairs for CrossEncoder.predict().
    # predict() expects a list of 2-tuples: (query_string, passage_string).
    # The cross-encoder tokenizes each pair as:
    #   [CLS] query [SEP] chunk_text [SEP]
    # and processes the full sequence together — joint attention.
    # This is why cross-encoders are more accurate than bi-encoders
    # for relevance scoring: the model sees both texts simultaneously.
    #
    # chunk["text"] is the Qdrant payload field name confirmed in
    # embedding_service.py: payload={"text": chunk.text, ...}

    scores = _reranker.predict(pairs)
    # predict() runs all pairs through the model in a single batch.
    # Returns a numpy array of float scores, one per pair.
    # Scores are raw logits from the model's classification head —
    # not probabilities, not bounded to [0,1]. Only relative ordering
    # matters. Higher = more relevant.
    # Running as a batch is faster than 20 individual calls: one
    # forward pass over the batched inputs (subject to max_length per pair).

    scored_chunks = sorted(
        zip(chunks, scores),
        key=lambda x: x[1],
        reverse=True,
    )
    # zip(chunks, scores) pairs each chunk dict with its score.
    # sorted(..., reverse=True) orders by score descending — most
    # relevant first.
    # sorted() returns a new list; the original chunks list is unchanged.
    # Alternative: numpy.argsort for index-based sorting — marginally
    # faster at 20 items but less readable. Not worth the complexity.

    top_chunks = [chunk for chunk, _ in scored_chunks[:top_k]]
    # Slice to top_k, unpack chunk from (chunk, score) tuple.
    # Score discarded — caller does not need it.

    logger.info(
        f"Reranked {len(chunks)} chunks → top {len(top_chunks)} selected. "
        f"Top score: {scored_chunks[0][1]:.4f}, "
        f"cutoff score: {scored_chunks[min(top_k - 1, len(scored_chunks) - 1)][1]:.4f}"
    )
    # Log: how many in, how many out, top score, score at the cut-off.
    # min(top_k - 1, len(scored_chunks) - 1) guards against the case
    # where fewer than top_k chunks were returned (e.g. corpus has only
    # 3 chunks total) — prevents index-out-of-bounds on the log line.
    # :.4f formats to 4 decimal places — sufficient for a relevance score.

    return top_chunks