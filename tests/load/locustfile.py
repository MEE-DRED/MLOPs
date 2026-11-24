"""Locust flood simulation:  pip install locust
    locust -f tests/load/locustfile.py --host http://127.0.0.1:8000 --headless -u 50 -r 10 -t 60s --csv outputs/reports/locust
Not executed in the build sandbox (Locust not installed). The dependency-free alternative is scripts/flood_test.py.
"""
import io

import numpy as np
from locust import HttpUser, between, task
from PIL import Image


def _png() -> bytes:
    yy, xx = np.mgrid[:224, :224]
    img = np.zeros((224, 224, 3), np.uint8)
    img[((yy - 112) ** 2 + (xx - 112) ** 2) < 100 ** 2] = (120, 60, 30)
    buf = io.BytesIO(); Image.fromarray(img).save(buf, "PNG"); return buf.getvalue()


PNG = _png()


class ApiUser(HttpUser):
    wait_time = between(0.1, 0.5)

    @task(5)
    def predict(self):
        self.client.post("/predict", files={"file": ("flood.png", PNG, "image/png")})

    @task(2)
    def health(self):
        self.client.get("/health")

    @task(1)
    def monitoring(self):
        self.client.get("/monitoring")
