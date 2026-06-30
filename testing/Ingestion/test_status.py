import urllib.request
import json

TOKEN       = "nTSNWUv9hHemknlIri6hkoxfZ9J_xPWN0YLo6eXLOGk"
DOCUMENT_ID = "e0a55c12-a4ef-4641-bf4f-9c17c4bf94ba"

req = urllib.request.Request(
    f"http://nginx/api/documents/{DOCUMENT_ID}/status",
    headers={"Authorization": f"Bearer {TOKEN}"},
)

try:
    resp = urllib.request.urlopen(req)
    print(json.dumps(json.loads(resp.read()), indent=2))
except urllib.error.HTTPError as e:
    print("Status:", e.code)
    print("Response:", e.read().decode())