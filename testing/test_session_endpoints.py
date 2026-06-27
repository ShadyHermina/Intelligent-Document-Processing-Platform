"""
testing/test_session_endpoints.py
----------------------------------
Runs all four session endpoint verifications against FastAPI directly.

How to run:
    docker cp testing/test_session_endpoints.py fastapi_a:/tmp/test_session_endpoints.py
    docker exec fastapi_a python /tmp/test_session_endpoints.py
"""

import json
import sys
import urllib.request
import urllib.error

BASE_URL = "http://localhost:8000"
SEED_PASSPHRASE = "correct-horse-battery-staple"
WRONG_PASSPHRASE = "wrong-passphrase"

PASSED = 0
FAILED = 0


def check(description: str, condition: bool) -> None:
    global PASSED, FAILED
    if condition:
        print(f"  PASS  {description}")
        PASSED += 1
    else:
        print(f"  FAIL  {description}")
        FAILED += 1


def post_json(path: str, body: dict) -> tuple[int, dict]:
    """Send a POST request with a JSON body. Returns (status_code, response_dict)."""
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        f"{BASE_URL}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode())


def get_json(path: str, headers: dict = {}) -> tuple[int, dict]:
    """Send a GET request with optional headers. Returns (status_code, response_dict)."""
    req = urllib.request.Request(
        f"{BASE_URL}{path}",
        headers=headers,
        method="GET",
    )
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode())


# ---------------------------------------------------------------------------
# Verification 2 — Correct passphrase returns 200 and a token
# ---------------------------------------------------------------------------

print()
print("=" * 60)
print("  POST /api/session/init")
print("=" * 60)

status, body = post_json("/api/session/init", {"access_phrase": SEED_PASSPHRASE})

print(f"  Status : {status}")
print(f"  Body   : {body}")

check("Correct passphrase returns HTTP 200", status == 200)
check("Response contains session_token field", "session_token" in body)

session_token = body.get("session_token", "")

check("session_token is 43 characters", len(session_token) == 43)
check(
    "session_token is opaque (not a JWT — no dots)",
    "." not in session_token
)
check(
    "session_token does not contain tenant name",
    "IDPP" not in session_token and "Test" not in session_token
)

print(f"\n  Token  : {session_token}")

# ---------------------------------------------------------------------------
# Verification 3 — Wrong passphrase returns 401
# ---------------------------------------------------------------------------

print()
print("=" * 60)
print("  POST /api/session/init — wrong passphrase")
print("=" * 60)

status_wrong, body_wrong = post_json(
    "/api/session/init",
    {"access_phrase": WRONG_PASSPHRASE}
)

print(f"  Status : {status_wrong}")
print(f"  Body   : {body_wrong}")

check("Wrong passphrase returns HTTP 401", status_wrong == 401)
check(
    "401 response body says 'unauthorized' only",
    body_wrong.get("detail") == "unauthorized"
)
check(
    "401 body does not mention passphrase or tenant",
    "passphrase" not in str(body_wrong).lower()
    and "tenant" not in str(body_wrong).lower()
    and "not found" not in str(body_wrong).lower()
)

# ---------------------------------------------------------------------------
# Verification 4 — Valid token resolves to tenant
# ---------------------------------------------------------------------------

print()
print("=" * 60)
print("  GET /api/session/me — valid token")
print("=" * 60)

if session_token:
    status_me, body_me = get_json(
        "/api/session/me",
        headers={"Authorization": f"Bearer {session_token}"}
    )

    print(f"  Status : {status_me}")
    print(f"  Body   : {body_me}")

    check("Valid token returns HTTP 200", status_me == 200)
    check("Response contains tenant_id", "tenant_id" in body_me)
    check("Response contains tenant_name", "tenant_name" in body_me)
    check(
        "tenant_name matches seed tenant",
        body_me.get("tenant_name") == "IDPP Test Tenant"
    )
    check(
        "tenant_id matches seed tenant",
        body_me.get("tenant_id") == "bd8c8de3-4a8e-48b9-9065-9ac08918a9c7"
    )
else:
    print("  SKIP — no token available (Verification 2 failed)")
    FAILED += 5

# ---------------------------------------------------------------------------
# Verification 5 — Invalid token returns 401
# ---------------------------------------------------------------------------

print()
print("=" * 60)
print("  GET /api/session/me — invalid token")
print("=" * 60)

status_bad, body_bad = get_json(
    "/api/session/me",
    headers={"Authorization": "Bearer invalidtokenthatshouldnotwork"}
)

print(f"  Status : {status_bad}")
print(f"  Body   : {body_bad}")

check("Invalid token returns HTTP 401", status_bad == 401)
check(
    "401 response body says 'unauthorized' only",
    body_bad.get("detail") == "unauthorized"
)

# ---------------------------------------------------------------------------
# Verification 6 — Missing Authorization header returns 401
# ---------------------------------------------------------------------------

print()
print("=" * 60)
print("  GET /api/session/me — missing Authorization header")
print("=" * 60)

status_missing, body_missing = get_json("/api/session/me")

print(f"  Status : {status_missing}")
print(f"  Body   : {body_missing}")

check("Missing Authorization header returns HTTP 401", status_missing == 401)

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

print()
print("=" * 60)
total = PASSED + FAILED
print(f"  Results: {PASSED}/{total} passed")
if FAILED == 0:
    print("  ALL TESTS PASSED")
else:
    print(f"  {FAILED} TEST(S) FAILED")
print("=" * 60)
print()

sys.exit(0 if FAILED == 0 else 1)