# test_nginx_routing.py
# Confirms that /api/health is reachable through Nginx on port 80.
# Run this inside any container on idp_network.
# Expected: HTTP 200 with a JSON body containing "status": "ok"

import urllib.request
import json

url = "http://nginx/api/health"

print(f"Hitting: {url}")

res = urllib.request.urlopen(url)

status = res.status
body = json.loads(res.read().decode())

print(f"HTTP Status : {status}")
print(f"Response    : {body}")

assert status == 200, f"Expected 200, got {status}"
assert body.get("status") == "ok", f"Expected status ok, got {body}"

print("\nPASS — Nginx is routing /api/health to FastAPI correctly")