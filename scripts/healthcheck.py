"""Container health check: exit 0 if the API answers /health with HTTP 200."""
import json
import os
import sys
import urllib.request

url = os.environ.get("DR_API_URL", "http://127.0.0.1:8000").rstrip("/") + "/health"
try:
    with urllib.request.urlopen(url, timeout=4) as r:
        body = json.loads(r.read())
        sys.exit(0 if r.status == 200 and "status" in body else 1)
except Exception:
    sys.exit(1)
