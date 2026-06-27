# fastapi/dependencies/auth.py
#
# Reusable FastAPI dependency for session token resolution.
#
# Any endpoint that needs to know which tenant is making a request
# declares this dependency instead of repeating token resolution logic.
#
# Usage in any endpoint:
#
#   from fastapi import Depends
#   from dependencies.auth import get_current_tenant, TenantContext
#
#   @router.post("/some-protected-endpoint")
#   async def my_endpoint(
#       request: Request,
#       tenant: TenantContext = Depends(get_current_tenant),
#   ):
#       # tenant.tenant_id and tenant.tenant_name are ready to use
#       # if the token was invalid, this function never runs — FastAPI
#       # already returned 401 from inside get_current_tenant

from dataclasses import dataclass

from fastapi import Request, HTTPException, status

from core.database import get_pool


# ---------------------------------------------------------------------------
# TenantContext
# ---------------------------------------------------------------------------

@dataclass
class TenantContext:
    """
    Carries resolved tenant identity from the auth dependency to endpoints.

    Produced by get_current_tenant() after successful token validation.
    Injected into endpoint handlers via FastAPI's Depends() mechanism.

    Why a dataclass and not a Pydantic model?

    Pydantic models are designed for serialization — converting to and from
    JSON, validating external input. TenantContext never crosses the network.
    It is an internal Python object that lives only for the duration of one
    request. A dataclass gives us typed attributes with zero overhead and
    no serialization machinery we do not need.

    Why not just return a tuple (tenant_id, tenant_name)?

    A tuple forces callers to use positional access: result[0], result[1].
    If the return order ever changes, every caller breaks silently.
    A dataclass uses named attributes: tenant.tenant_id, tenant.tenant_name.
    The names document intent and are robust to future additions.
    """
    tenant_id: str
    tenant_name: str


# ---------------------------------------------------------------------------
# Dependency
# ---------------------------------------------------------------------------

async def get_current_tenant(request: Request) -> TenantContext:
    """
    FastAPI dependency that resolves a Bearer token to a TenantContext.

    Called automatically by FastAPI before any endpoint that declares:
        tenant: TenantContext = Depends(get_current_tenant)

    If the token is missing, malformed, expired, or unknown, this function
    raises HTTP 401 and the endpoint handler never runs. FastAPI propagates
    the exception directly to the client.

    If the token is valid, this function returns a TenantContext containing
    the resolved tenant_id and tenant_name. FastAPI injects this object
    into the endpoint handler as the value of the tenant parameter.

    Why async def?

    This function queries PostgreSQL — a network I/O operation that must
    be awaited. FastAPI handles async dependencies correctly, awaiting
    them before calling the endpoint handler.

    Parameters
    ----------
    request : Request
        The FastAPI request object, injected automatically by FastAPI
        when this function is used as a dependency. Contains headers,
        the app instance (for pool access), and client information.

    Returns
    -------
    TenantContext
        Resolved tenant identity. Guaranteed non-null if returned —
        the function either returns a valid context or raises 401.

    Raises
    ------
    HTTPException (401)
        If the Authorization header is missing, malformed, or contains
        a token that is expired, unknown, or belongs to an inactive tenant.
        The detail is always "unauthorized" — no information about which
        specific condition failed is revealed to the caller.
    """

    # ---- Step 1: Extract token from Authorization header ----

    auth_header = request.headers.get("authorization", "")
    # FastAPI lowercases all incoming header names.
    # "Authorization: Bearer ..." arrives as "authorization" in the dict.
    # Default to empty string so startswith() below is always safe.

    if not auth_header.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="unauthorized",
        )

    token = auth_header[len("Bearer "):]
    # Slice off the 7-character "Bearer " prefix.
    # What remains is the raw 43-character opaque token.

    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="unauthorized",
        )
    # Guard against "Bearer " with nothing after it.
    # Unlikely from a real client but worth closing the gap.

    # ---- Step 2: Resolve token to tenant via PostgreSQL ----

    pool = get_pool(request.app)
    # get_pool() is synchronous — it just retrieves app.state.db_pool.
    # No await needed here.

    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT t.id, t.name
            FROM sessions s
            JOIN tenants t ON t.id = s.tenant_id
            WHERE s.token    = $1
              AND s.expires_at > now()
              AND t.is_active  = TRUE
            """,
            token,
        )
        # Single query resolves all three conditions simultaneously:
        #   - Token exists in sessions table
        #   - Token has not expired
        #   - The tenant it belongs to is active
        #
        # If any condition fails, row is None.
        # The client cannot tell which condition failed.

    if row is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="unauthorized",
        )

    # ---- Step 3: Return resolved identity ----

    return TenantContext(
        tenant_id=str(row["id"]),
        tenant_name=row["name"],
    )
    # str(row["id"]) converts the asyncpg UUID object to a plain string.
    # All downstream code works with string UUIDs — consistent with how
    # we handle UUIDs everywhere else in the application.