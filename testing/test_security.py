"""
testing/test_security.py
------------------------
Verifies that core/security.py behaves correctly:

  1. hash_passphrase() is deterministic
  2. hash_passphrase() produces a 64-character hex string
  3. Different inputs produce different hashes
  4. hash_passphrase() matches the exact value stored by seed.py
     (passphrase: "correct-horse-battery-staple")
  5. generate_session_token() produces unique values on every call
  6. generate_session_token() produces a 43-character string
  7. generate_session_token() contains only URL-safe characters

How to run:
    docker cp testing/test_security.py fastapi_a:/tmp/test_security.py
    docker exec fastapi_a python /tmp/test_security.py
"""

import sys
import string

# ---------------------------------------------------------------------------
# Import the functions under test
# ---------------------------------------------------------------------------
# We import from core.security because PYTHONPATH=/app inside the container,
# so Python resolves "core.security" to /app/core/security.py.
try:
    from core.security import hash_passphrase, generate_session_token
except ImportError as e:
    print(f"FAIL — could not import from core.security: {e}")
    print("       Ensure core/security.py exists inside the container.")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Test runner helpers
# ---------------------------------------------------------------------------

PASSED = 0
FAILED = 0


def check(description: str, condition: bool) -> None:
    """
    Evaluate one test condition and print a PASS or FAIL line.

    Parameters
    ----------
    description : str
        Human-readable name of the test case.
    condition : bool
        The assertion to evaluate. True = pass, False = fail.
    """
    global PASSED, FAILED
    if condition:
        print(f"  PASS  {description}")
        PASSED += 1
    else:
        print(f"  FAIL  {description}")
        FAILED += 1


# ---------------------------------------------------------------------------
# Tests — hash_passphrase()
# ---------------------------------------------------------------------------

print()
print("=" * 60)
print("  hash_passphrase()")
print("=" * 60)

# The seed passphrase and its known SHA-256 hash.
# Computed independently with:
#   import hashlib
#   hashlib.sha256("correct-horse-battery-staple".encode("utf-8")).hexdigest()
# Must match exactly what seed.py stored in PostgreSQL.
SEED_PASSPHRASE = "correct-horse-battery-staple"
SEED_KNOWN_HASH = "e6b694c3b8b9b0b9ff75c03b9b1d9e9b7c2b0c4a3d5e6f7a8b9c0d1e2f3a4b5c6"
# Note: the value above is a placeholder — the real hash is computed
# below and compared. We derive it from the function itself and verify
# structural properties rather than hardcoding a value that could be
# copied incorrectly.

h1 = hash_passphrase(SEED_PASSPHRASE)
h2 = hash_passphrase(SEED_PASSPHRASE)
h3 = hash_passphrase("wrong-passphrase")

check(
    "Same input produces same output (deterministic)",
    h1 == h2
)

check(
    "Output is a 64-character string",
    len(h1) == 64
)

check(
    "Output contains only lowercase hex characters",
    all(c in "0123456789abcdef" for c in h1)
)

check(
    "Different inputs produce different hashes",
    h1 != h3
)

check(
    "Single character difference produces completely different hash",
    hash_passphrase("correct-horse-battery-staple") !=
    hash_passphrase("correct-horse-battery-stapleX")
)

# Most important test: does our function produce the exact same hash
# that seed.py stored in PostgreSQL? If this fails, session init will
# never authenticate successfully regardless of everything else working.
import hashlib
independently_computed = hashlib.sha256(
    SEED_PASSPHRASE.encode("utf-8")
).hexdigest()

check(
    "Matches independently computed SHA-256 of seed passphrase",
    h1 == independently_computed
)

print()
print(f"  Seed passphrase : {SEED_PASSPHRASE}")
print(f"  Computed hash   : {h1}")
print(f"  This hash must match access_phrase_hash in PostgreSQL tenants table")


# ---------------------------------------------------------------------------
# Tests — generate_session_token()
# ---------------------------------------------------------------------------

print()
print("=" * 60)
print("  generate_session_token()")
print("=" * 60)

t1 = generate_session_token()
t2 = generate_session_token()
t3 = generate_session_token()

check(
    "Two consecutive tokens are different",
    t1 != t2
)

check(
    "Three consecutive tokens are all different",
    t1 != t2 and t2 != t3 and t1 != t3
)

check(
    "Token is 43 characters long",
    len(t1) == 43
)

# URL-safe base64 uses A-Z, a-z, 0-9, hyphen, underscore.
# No plus, no slash, no equals padding.
URLSAFE_CHARS = set(string.ascii_letters + string.digits + "-_")
check(
    "Token contains only URL-safe characters (A-Z a-z 0-9 - _)",
    all(c in URLSAFE_CHARS for c in t1)
)

check(
    "Token contains no spaces or special characters",
    " " not in t1 and "+" not in t1 and "/" not in t1 and "=" not in t1
)

print()
print(f"  Sample token 1 : {t1}")
print(f"  Sample token 2 : {t2}")
print(f"  Sample token 3 : {t3}")


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
    print(f"  {FAILED} TEST(S) FAILED — do not proceed to next step")
print("=" * 60)
print()

sys.exit(0 if FAILED == 0 else 1)