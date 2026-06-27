# test_nginx_rate_limiting.py
# Sends 60 simultaneous requests to Nginx.
# Config: rate=10r/s, burst=20, nodelay, limit_req_status=429
#
# All 60 requests arrive within the same 100ms window.
# Nginx has 1 base rate slot + 20 burst slots = 21 total slots available.
#
# Expected outcome:
#   21 requests get HTTP 200  (1 base rate slot + 20 burst slots)
#   39 requests get HTTP 429  (no slot available, rejected immediately)

import urllib.request
import urllib.error
import threading

TOTAL_REQUESTS  = 60
EXPECTED_200    = 21
EXPECTED_429    = 39

url = "http://nginx/api/health"
results = []
lock = threading.Lock()
barrier = threading.Barrier(TOTAL_REQUESTS)

def send_request(i):
    barrier.wait()
    try:
        res = urllib.request.urlopen(url, timeout=5)
        code = res.status
    except urllib.error.HTTPError as e:
        code = e.code
    except Exception:
        code = "error"

    with lock:
        results.append(code)
        print(f"Request {i:02d} → HTTP {code}")

threads = []
for i in range(1, TOTAL_REQUESTS + 1):
    t = threading.Thread(target=send_request, args=(i,))
    threads.append(t)

print(f"Firing {TOTAL_REQUESTS} simultaneous requests to {url}\n")

for t in threads:
    t.start()

for t in threads:
    t.join()

count_200   = results.count(200)
count_429   = results.count(429)
count_other = [r for r in results if r not in (200, 429)]

print(f"\nSummary:")
print(f"  200 OK           : {count_200}  (expected {EXPECTED_200})")
print(f"  429 Rate Limited : {count_429}  (expected {EXPECTED_429})")
print(f"  Other            : {count_other}")

assert count_200 == EXPECTED_200, f"Expected exactly {EXPECTED_200} successful requests, got {count_200}"
assert count_429 == EXPECTED_429, f"Expected exactly {EXPECTED_429} rate-limited requests, got {count_429}"

print("\nPASS — Rate limiter is working exactly as configured")