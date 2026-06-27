# fastapi/routers/session.py
#
# Session management endpoints.
#
# POST /init  — resolve passphrase to tenant, issue session token
# GET  /me    — resolve session token to tenant name
#
# These two endpoints are the security foundation of the platform.
# Every other endpoint in every future phase will depend on the
# session token issued here.

from fastapi import APIRouter, Depends, Request, HTTPException, status
from pydantic import BaseModel

from core.database import get_pool
from core.security import hash_passphrase, generate_session_token
from core.config import get_settings
from dependencies.auth import get_current_tenant, TenantContext


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

router = APIRouter()
# APIRouter() creates a router with no prefix — the prefix "/api/session"
# is applied in main.py when this router is registered with
# app.include_router(router, prefix="/api/session").
# Keeping the prefix out of this file means the router is portable —
# it could be mounted at a different prefix without changing this file.


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class SessionInitRequest(BaseModel):
    """
    Request body for POST /api/session/init.

    Pydantic validates the incoming JSON body against this model.
    If access_phrase is missing or not a string, FastAPI returns
    HTTP 422 automatically before our handler function runs.
    We write zero defensive code for malformed input.
    """
    access_phrase: str
    # str type means any non-null string is accepted at the Pydantic level.
    # We do not validate length or format here — an empty string will simply
    # produce a hash that matches nothing in the database, which returns 401.
    # No special case needed.


class SessionInitResponse(BaseModel):
    """
    Response body for a successful POST /api/session/init.

    Only the session token is returned. The tenant_id, tenant name,
    and expiry time are deliberately excluded — the token is opaque
    and the client needs to know nothing about what it represents.
    """
    session_token: str


class SessionMeResponse(BaseModel):
    """
    Response body for GET /api/session/me.

    Returns enough information for the frontend to personalize the UI
    (tenant name) and for the frontend to identify the tenant in
    subsequent API calls (tenant_id as a string).
    """
    tenant_id: str
    tenant_name: str


# ---------------------------------------------------------------------------
# Helper — extract client IP
# ---------------------------------------------------------------------------

def get_client_ip(request: Request) -> str:
    """
    Extract the client IP address from the request for audit logging.

    When a request passes through Nginx, the original client IP is
    placed in the X-Forwarded-For header by Nginx. The request.client
    attribute would give us the Nginx container's IP — not the real
    client. We prefer X-Forwarded-For when present.

    Returns "unknown" rather than raising if the IP cannot be determined —
    a missing IP should never block a legitimate request.
    """
    forwarded_for = request.headers.get("x-forwarded-for")
    if forwarded_for:
        # X-Forwarded-For can contain a comma-separated list when multiple
        # proxies are in the chain: "client, proxy1, proxy2".
        # The leftmost value is always the original client IP.
        return forwarded_for.split(",")[0].strip()

    if request.client:
        return request.client.host

    return "unknown"


# ---------------------------------------------------------------------------
# POST /init
# ---------------------------------------------------------------------------

@router.post(
    "/init",
    response_model=SessionInitResponse,
    status_code=status.HTTP_200_OK,
)
async def session_init(body: SessionInitRequest, request: Request):
    """
    Resolve a passphrase to a tenant and issue a session token.

    Flow:
      1. Hash the incoming passphrase
      2. Look up the hash in the tenants table
      3. On miss: write failed audit log entry, return 401
      4. On hit: generate token, store in sessions, write success audit log
      5. Return the token

    Security properties:
      - Plaintext passphrase never touches the database
      - Wrong passphrase and unknown tenant return identical 401 response
      - Timing difference between hit and miss is minimal (one vs two queries)
      - Audit log records all attempts, successful and failed
    """
    settings = get_settings()
    pool = get_pool(request.app)
    client_ip = get_client_ip(request)

    # Step 1 — hash the passphrase
    phrase_hash = hash_passphrase(body.access_phrase)
    # body.access_phrase is the plaintext string from the request body.
    # After this line, we never use body.access_phrase again.
    # phrase_hash is what we send to PostgreSQL.

    async with pool.acquire() as conn:
        # pool.acquire() borrows one connection from the pool.
        # async with ensures the connection is returned to the pool
        # when this block exits — whether normally or via exception.
        # All three database operations (lookup, insert session, insert
        # audit) happen on the same connection inside this block.

        # Step 2 — look up the hash
        row = await conn.fetchrow(
            """
            SELECT id, name
            FROM tenants
            WHERE access_phrase_hash = $1
              AND is_active = TRUE
            """,
            phrase_hash,
        )
        # conn.fetchrow() returns one asyncpg Record or None.
        # $1 is the parameterized placeholder — asyncpg substitutes
        # phrase_hash safely, preventing SQL injection.
        # We never use f-strings or string concatenation for SQL values.
        #
        # The single query approach: we do NOT first check if the tenant
        # exists and then check the passphrase in a second query.
        # A two-query approach leaks timing information — an attacker
        # can measure whether "tenant not found" is faster than
        # "tenant found, wrong passphrase" and use that to enumerate
        # valid tenant names. One query collapses both failure modes.

        if row is None:
            # Step 3 — failed attempt: write audit log, return 401
            await conn.execute(
                """
                INSERT INTO audit_log
                    (tenant_id, actor, action, target_type)
                VALUES
                    (NULL, $1, $2, $3)
                """,
                client_ip,
                "session_init_failed",
                "session",
            )
            # tenant_id is NULL — we do not know who attempted this.
            # actor is the client IP — the only identity we have.
            # We commit this before raising the exception so the audit
            # record is never lost even if something fails afterward.

            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="unauthorized",
            )
            # "unauthorized" is intentionally vague.
            # It does not say "wrong passphrase" or "tenant not found".
            # Both failure modes return this identical response body.
            # An attacker learns nothing from the response.

        # Row was found — extract tenant data
        tenant_id: str = str(row["id"])
        tenant_name: str = row["name"]
        # row["id"] is a Python UUID object from asyncpg.
        # We convert it to str immediately — all our code works with
        # string UUIDs, and PostgreSQL $1 parameters accept str fine.

        # Step 4a — generate the session token
        token = generate_session_token()
        # 43-character URL-safe random string. Zero information about
        # the tenant encoded inside it.

        # Step 4b — store the session in PostgreSQL
        await conn.execute(
            """
            INSERT INTO sessions (token, tenant_id, expires_at)
            VALUES ($1, $2, now() + ($3 || ' hours')::interval)
            """,
            token,
            tenant_id,
            str(settings.session_ttl_hours),
        )
        # $1 = the opaque token string
        # $2 = tenant UUID as string
        # $3 = session TTL in hours (default 8, from config)
        #
        # now() + ($3 || ' hours')::interval builds the expiry timestamp.
        # ($3 || ' hours') concatenates the number with the word "hours"
        # to form a valid PostgreSQL interval string like "8 hours".
        # ::interval casts it to PostgreSQL's interval type.
        # now() + interval '8 hours' gives us the expiry timestamp.

        # Step 4c — write success audit log
        await conn.execute(
            """
            INSERT INTO audit_log
                (tenant_id, actor, action, target_type)
            VALUES
                ($1, $2, $3, $4)
            """,
            tenant_id,
            client_ip,
            "session_init_success",
            "session",
        )
        # tenant_id is now known — we record who authenticated.
        # This row, combined with the failed-attempt rows, gives operators
        # a complete picture of authentication activity per tenant.

    # Step 5 — return the token
    return SessionInitResponse(session_token=token)
    # Only the token is returned. tenant_id and tenant_name stay
    # on the server. The client has a random string that means nothing
    # without the sessions table to look it up in.


# ---------------------------------------------------------------------------
# GET /me
# ---------------------------------------------------------------------------

@router.get(
    "/me",
    response_model=SessionMeResponse,
    status_code=status.HTTP_200_OK,
)
async def session_me(
    request: Request,
    tenant: TenantContext = Depends(get_current_tenant),
):
    """
    Resolve a session token to the tenant it represents.

    Token extraction and validation is handled entirely by the
    get_current_tenant dependency. If this handler runs, the token
    is already validated and tenant contains the resolved identity.

    The client sends the token in the Authorization header:
        Authorization: Bearer <token>
    """
    return SessionMeResponse(
        tenant_id=tenant.tenant_id,
        tenant_name=tenant.tenant_name,
    )