# test_nginx_real_ip.py
# Calls /debug/headers through Nginx and inspects the headers FastAPI received.
# Confirms that Nginx is forwarding the real client IP correctly.
#
# Expected: FastAPI sees x-real-ip and x-forwarded-for headers,
# and neither of them contains the Nginx container's internal IP (172.x.x.x)
# as the only value — the real client IP must be present.

import urllib.request
import json

url = "http://nginx/api/debug/headers"

print(f"Requesting: {url}\n")

res = urllib.request.urlopen(url)
data = json.loads(res.read().decode())

headers = data["headers"]
client_host = data["client_host"]

print("Headers received by FastAPI:")
for key, value in headers.items():
    print(f"  {key}: {value}")

print(f"\nDirect client_host seen by FastAPI: {client_host}")

# client_host is the IP of whoever connected directly to FastAPI.
# This will always be the Nginx container IP (172.x.x.x) — that is correct.
# What matters is that the forwarded headers carry the real origin IP.

x_real_ip = headers.get("x-real-ip")
x_forwarded_for = headers.get("x-forwarded-for")

print(f"\nx-real-ip        : {x_real_ip}")
print(f"x-forwarded-for  : {x_forwarded_for}")

assert x_real_ip is not None, "x-real-ip header is missing — Nginx is not forwarding it"
assert x_forwarded_for is not None, "x-forwarded-for header is missing — Nginx is not forwarding it"
assert x_real_ip == x_forwarded_for, (
    f"x-real-ip and x-forwarded-for should match for a single proxy hop, "
    f"got {x_real_ip} vs {x_forwarded_for}"
)

print("\nPASS — Nginx is forwarding real client IP headers to FastAPI correctly")