import urllib.request
import json
import uuid

TOKEN = "nTSNWUv9hHemknlIri6hkoxfZ9J_xPWN0YLo6eXLOGk"
FILE_PATH = "/tmp/test_document.pdf"

boundary = b"----FormBoundary" + str(uuid.uuid4()).replace("-", "").encode()[:16]

with open(FILE_PATH, "rb") as f:
    file_bytes = f.read()

body = (
    b"--" + boundary + b"\r\n"
    b'Content-Disposition: form-data; name="file"; filename="test_document.pdf"\r\n'
    b"Content-Type: application/pdf\r\n"
    b"\r\n"
    + file_bytes
    + b"\r\n--" + boundary + b"--\r\n"
)

req = urllib.request.Request(
    "http://nginx/api/documents/upload",
    data=body,
    headers={
        "Authorization": f"Bearer {TOKEN}",
        "Content-Type": "multipart/form-data; boundary=" + boundary.decode(),
    },
    method="POST",
)

try:
    resp = urllib.request.urlopen(req)
    print(json.dumps(json.loads(resp.read()), indent=2))
except urllib.error.HTTPError as e:
    print("Status:", e.code)
    print("Response:", e.read().decode())