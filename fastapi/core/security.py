# fastapi/core/security.py
#
# Pure cryptographic operations with no external dependencies beyond
# Python's standard library. No database calls. No HTTP. No app state.
#
# Two responsibilities:
#   1. hash_passphrase(phrase)     — SHA-256 hash for DB comparison
#   2. generate_session_token()    — cryptographically random opaque token
#
# Both functions are synchronous — they perform CPU-bound computation
# with no I/O waiting. There is nothing to await, so async def would
# be misleading and add unnecessary overhead. FastAPI can call
# synchronous functions from async endpoints without any problem —
# it only needs await when the function itself does I/O.

import hashlib
import secrets


def hash_passphrase(phrase: str) -> str:
    """
    Return the SHA-256 hex digest of the given passphrase string.

    This is a one-way transformation. The same input always produces
    the same output. The output cannot be reversed to recover the input.

    We compare hash-to-hash rather than plaintext-to-plaintext:
      - Client sends plaintext passphrase in the request body
      - We hash it here
      - We query PostgreSQL: WHERE access_phrase_hash = $1
      - PostgreSQL compares our hash against the stored hash
      - The plaintext never touches the database

    Why UTF-8 encoding before hashing?

    hashlib.sha256() requires bytes, not a str. encode("utf-8") converts
    the Python string to bytes using the UTF-8 encoding. UTF-8 is chosen
    because it is the universal standard for text encoding and handles
    all Unicode characters correctly. Using the system default encoding
    (which encode() uses without an argument) would produce inconsistent
    results across different operating systems.

    Parameters
    ----------
    phrase : str
        The plaintext passphrase as received from the client.

    Returns
    -------
    str
        64-character lowercase hex string — the SHA-256 hash.
        Example: "9f86d081884c7d659a2feaa0c55ad015a3bf4f1b748fd7d5d3ce6b4e7db8a6f2"

    Example
    -------
    >>> hash_passphrase("acme-secret-2024")
    "9f86d081884c7d659a2feaa0c55ad015a3bf4f1b748fd7d5d3ce6b4e7db8a6f2"
    """
    encoded = phrase.encode("utf-8")
    # phrase is a Python str — a sequence of Unicode characters.
    # encode("utf-8") converts it to bytes — a sequence of raw bytes.
    # SHA-256 operates on bytes, not strings.
    # Example: "abc" → b"abc" → bytes 0x61, 0x62, 0x63

    digest = hashlib.sha256(encoded)
    # hashlib.sha256() creates a hash object initialized with our bytes.
    # The hash object is not the hash value yet — it is a stateful object
    # that can accept more data via .update() if needed (we do not need
    # that here — our entire input is in one call).

    return digest.hexdigest()
    # hexdigest() finalizes the hash and returns it as a 64-character
    # lowercase hex string. Each byte of the 32-byte SHA-256 output
    # is represented as two hex characters: 32 bytes × 2 chars = 64 chars.
    # Alternative: digest() returns the raw 32 bytes. We use hexdigest()
    # because hex strings are safe to store in PostgreSQL TEXT columns
    # and to compare with == without encoding concerns.


def generate_session_token() -> str:
    """
    Generate a cryptographically secure random session token.

    The token is opaque — it encodes no information about the tenant,
    the session expiry, or any other application state. Its only meaning
    comes from the row in the sessions table that maps it to a tenant_id.

    Why secrets and not random?

    Python's random module uses a Mersenne Twister algorithm seeded from
    the current time. It is designed for simulations and games — situations
    where statistical randomness is needed but security is not. Given the
    seed (the timestamp), an attacker can predict all future outputs.

    secrets uses the operating system's CSPRNG:
      - On Linux/macOS: /dev/urandom (kernel entropy pool)
      - On Windows: CryptGenRandom (Windows cryptographic API)
    Both draw entropy from physical hardware sources that are impossible
    to predict. No seed is exposed. No output can be used to predict
    future outputs.

    Why 32 bytes?

    32 bytes = 256 bits of entropy.
    The probability of two tokens colliding is 1 in 2^256 —
    a number larger than the atoms in the observable universe.
    The probability of an attacker guessing a valid token by brute
    force is equally negligible — even testing a billion tokens per
    second for a billion years would not produce a match.

    Why token_urlsafe and not token_hex?

    token_hex(32) produces a 64-character hex string using only 0-9 and a-f.
    token_urlsafe(32) produces a 43-character base64 string using
    A-Z, a-z, 0-9, - and _. token_urlsafe has more characters per byte
    of output (43 vs 64 chars for the same 32 bytes of entropy), is safe
    to use directly in HTTP Authorization headers and URLs without
    percent-encoding, and is the documented standard for session tokens
    in Python's own security documentation.

    Returns
    -------
    str
        43-character URL-safe base64 string.
        Example: "j-6XpTqRmN2oKwL8dVsEyBhCgIuAfZ0Pe1xYt3Mn4Qk"
    """
    return secrets.token_urlsafe(32)
    # token_urlsafe(32) asks the OS for 32 random bytes, then encodes
    # them using URL-safe base64 (RFC 4648).
    # Base64 encodes 3 bytes as 4 characters: 32 bytes → ~43 characters.
    # The = padding that standard base64 uses is stripped, which is why
    # the result is 43 characters rather than 44.