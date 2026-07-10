# fastapi/core/config.py
#
# Single source of truth for all application configuration.
# Every environment variable the app needs is declared here as a typed field.
# No other file in the application calls os.getenv() or os.environ directly.
#
# How pydantic-settings resolves values (in priority order):
#   1. Environment variables already present in the process (injected by Docker)
#   2. Values read from the .env file specified in class Config below
#   3. Field defaults declared in this class
#
# Inside Docker: step 1 always wins. The .env file is never read.
# Outside Docker (tests, local scripts): step 2 loads the .env file.
# Fields with no match in either source and no default → ValidationError at startup.

from functools import lru_cache
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """
    Typed application settings loaded from environment variables.

    pydantic-settings matches environment variable names to field names
    case-insensitively. POSTGRES_HOST in the environment maps to the
    postgres_host field below. The underscore/uppercase convention in
    environment variables is standard; lowercase with underscores is
    standard Python. pydantic-settings bridges the two automatically.

    Fields without defaults are REQUIRED. If they are absent from both
    the environment and the .env file, pydantic raises a ValidationError
    at startup, naming the missing field explicitly. This is intentional:
    a misconfigured application fails immediately at startup rather than
    failing silently on the first request that needs the missing value.
    """

    # ------------------------------------------------------------------
    # PostgreSQL
    # POSTGRES_HOST and POSTGRES_PORT are hardcoded in docker-compose.yml
    # (not in .env) because they are structural facts about the Docker
    # network, not deployment secrets. They are still declared here as
    # required fields because the application cannot function without them.
    # ------------------------------------------------------------------
    postgres_user: str
    postgres_password: str
    postgres_db: str
    postgres_host: str
    postgres_port: int
    # int declaration causes pydantic-settings to call int("5432").
    # If the value cannot be coerced to int, startup fails with a clear
    # ValidationError rather than a cryptic TypeError later at connection time.

    # ------------------------------------------------------------------
    # Qdrant
    # QDRANT_HOST and QDRANT_PORT come from .env.
    # QDRANT_PORT arrives as the string "6333" and is coerced to int.
    # ------------------------------------------------------------------
    qdrant_host: str
    qdrant_port: int

    # ------------------------------------------------------------------
    # OpenAI
    # OPENAI_API_KEY comes from .env.
    # Declared as required — the platform cannot process documents without it.
    # ------------------------------------------------------------------
    openai_api_key: str

    # ------------------------------------------------------------------
    # Application metadata
    # Both have defaults because they are labels, not secrets.
    # INSTANCE_ID is injected by docker-compose but is absent from .env —
    # the default "unknown" prevents a ValidationError outside Docker.
    # APP_ENV and LOG_LEVEL are in .env with values "development" and "info".
    # ------------------------------------------------------------------
    instance_id: str = "unknown"
    app_env: str = "development"
    log_level: str = "info"

    # ------------------------------------------------------------------
    # Session management
    # SESSION_TTL_HOURS is not in .env — it never needs to change per
    # deployment. The default of 8 covers a standard working day.
    # Adding SESSION_TTL_HOURS to docker-compose or .env overrides it
    # without touching code.
    # ------------------------------------------------------------------
    session_ttl_hours: int = 8

    # ------------------------------------------------------------------
    # Chunking — Phase 6
    # CHUNK_SIMILARITY_THRESHOLD controls Level 2 of the three-level
    # chunking pipeline. It is the cosine similarity score below which
    # two adjacent sentences are considered to have drifted to a new
    # topic, triggering a chunk boundary cut.
    #
    # Range:  0.0 (cut everywhere) to 1.0 (never cut)
    # Default 0.55: empirically reasonable for mixed enterprise documents.
    #   Lower values (e.g. 0.4) → larger chunks, more topic mixing allowed
    #   Higher values (e.g. 0.7) → smaller chunks, stricter topic coherence
    #
    # Tunable without code changes: add
    #   CHUNK_SIMILARITY_THRESHOLD=0.6
    # to docker-compose.yml under fastapi_a and fastapi_b environment
    # blocks, then restart those containers.
    # ------------------------------------------------------------------
    chunk_similarity_threshold: float = 0.55

    # ------------------------------------------------------------------
    # Phase 9 — RAG chat pipeline
    # RERANKER_MODEL: HuggingFace model identifier for the cross-encoder.
    #   Loaded once at startup via load_reranker() in lifespan.
    #   Changing this value requires a container rebuild only if the new
    #   model is not already cached — otherwise a restart suffices.
    # CHAT_HISTORY_LIMIT: maximum number of message objects kept in the
    #   in-memory conversation history per WebSocket connection.
    #   10 objects = 5 turns (user + assistant pairs).
    #   Prevents context window overflow on long sessions.
    # TOP_K_RETRIEVAL: how many candidate chunks Qdrant returns per query.
    #   Must match the top_k sent to query_knowledge_base FastMCP tool.
    #   20 is the value locked in Phase 8 design.
    # TOP_K_RERANK: how many chunks survive reranking and are passed to
    #   the LLM as context. 5 gives ~1500 tokens of context at 300
    #   tokens per chunk — substantial signal without dominating the
    #   GPT-4o context window.
    # ------------------------------------------------------------------
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    chat_history_limit: int = 10
    top_k_retrieval: int = 20
    top_k_rerank: int = 5

    class Config:
        # env_file: where to look for a .env file when environment variables
        # are not already present in the process. This path is relative to
        # wherever the Python process is launched from.
        # Inside Docker: /app is the working directory (set in Dockerfile),
        # but .env is not copied into the container — Docker injects the
        # variables directly, so this line is never actually used inside Docker.
        # Outside Docker: the process is launched from the project root where
        # .env lives, so this resolves correctly.
        env_file = ".env"
        env_file_encoding = "utf-8"

        # extra = "ignore": silently skip any environment variable not declared
        # as a field above. Without this, PATH, HOSTNAME, PYTHONPATH and every
        # other variable Docker injects would cause a ValidationError.
        # Alternative: extra = "forbid" — fails on any undeclared variable.
        # Useful in strict production audits to catch variable name typos.
        # We use "ignore" because Docker's environment contains many variables
        # we did not put there and do not want to enumerate.
        extra = "ignore"


@lru_cache
def get_settings() -> Settings:
    """
    Return the single application-wide Settings instance.

    lru_cache makes this function execute exactly once per Python process.
    The first call constructs the Settings object: reads all environment
    variables, coerces types, validates required fields, raises on error.
    Every subsequent call returns the cached object at zero cost.

    Why lru_cache and not a module-level singleton?

    A module-level `settings = Settings()` runs the moment this module is
    imported. If a test needs to override an environment variable before
    Settings is created, it is already too late — the import already fired.
    lru_cache defers construction to the first get_settings() call, giving
    test setup code time to patch the environment first.

    Why not call Settings() directly at each use site?

    Settings() re-reads and re-validates every environment variable on each
    call. Under load this wastes CPU. More importantly, it means two parts
    of the application could theoretically see different values if an
    environment variable changed between calls — an impossible but confusing
    scenario. One cached instance eliminates both problems.
    """
    return Settings()