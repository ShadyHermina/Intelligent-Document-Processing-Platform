# test_nginx_load_balancing.py
# Sends 10 sequential requests to Nginx and records which FastAPI instance
# handled each one. Confirms round-robin load balancing is working.
# 10 requests stays safely under the rate limit (21 allowed per burst).
# Expected: both fastapi_a and fastapi_b receive traffic.

import urllib.request
import json

url = "http://nginx/api/health"
results = []

print(f"Sending 10 requests to {url}\n")

for i in range(1, 11):
    res = urllib.request.urlopen(url)
    body = json.loads(res.read().decode())
    instance = body.get("instance", "unknown")
    results.append(instance)
    print(f"Request {i:02d} → {instance}")

print(f"\nSequence: {' → '.join(results)}")

unique = set(results)
assert len(unique) == 2, f"Expected both instances to respond, got: {unique}"
print("\nPASS — Both FastAPI instances received traffic from Nginx")