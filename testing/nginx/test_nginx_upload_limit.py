# test_nginx_upload_limit.py
# Sends two requests to Nginx:
#   1. A 1MB body  → should pass through (under 20MB limit)
#   2. A 21MB body → should be rejected immediately by Nginx with HTTP 413
#
# The 413 response confirms Nginx is enforcing client_max_body_size 20m
# before the request reaches FastAPI.

import urllib.request
import urllib.error

def send_request(label, url, data):
    try:
        req = urllib.request.Request(url, data=data, method="POST")
        req.add_header("Content-Type", "application/octet-stream")
        res = urllib.request.urlopen(req, timeout=30)
        return res.status
    except urllib.error.HTTPError as e:
        return e.code
    except Exception as e:
        return f"error: {e}"

url = "http://nginx/api/health"

# 1MB of zeros
small_body = b"\x00" * (1 * 1024 * 1024)

# 21MB of zeros — exceeds the 20MB limit
large_body = b"\x00" * (21 * 1024 * 1024)

print("Test 1 — 1MB request (should be allowed)")
code = send_request("1MB", url, small_body)
print(f"  HTTP {code}  (expected 200 or 405)\n")

print("Test 2 — 21MB request (should be rejected by Nginx)")
code = send_request("21MB", url, large_body)
print(f"  HTTP {code}  (expected 413)\n")

assert code == 413, f"Expected 413 for oversized request, got {code}"
print("PASS — Nginx is rejecting oversized uploads with HTTP 413")