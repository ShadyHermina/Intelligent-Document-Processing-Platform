import urllib.request
import json
import uuid

TOKEN = "nTSNWUv9hHemknlIri6hkoxfZ9J_xPWN0YLo6eXLOGk"

boundary = b"----FormBoundary" + str(uuid.uuid4()).replace("-", "").encode()[:16]

body = (
    b"--" + boundary + b"\r\n"
    b'Content-Disposition: form-data; name="file"; filename="test.txt"\r\n'
    b"Content-Type: text/plain\r\n"
    b"\r\n"
    b"This is a plain text file.\r\n"
    b"--" + boundary + b"--\r\n"
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
    urllib.request.urlopen(req)
except urllib.error.HTTPError as e:
    print("Status:", e.code)
    print("Response:", e.read().decode())