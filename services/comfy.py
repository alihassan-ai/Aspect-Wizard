import requests

class Comfy:
    def __init__(self, base: str, client_id: str):
        self.base = base
        self.client_id = client_id

    def get(self, path: str, timeout=20):
        r = requests.get(f"{self.base}{path}", timeout=timeout)
        r.raise_for_status()
        return r.json()

    def post(self, path: str, payload: dict, timeout=60):
        r = requests.post(f"{self.base}{path}", json=payload, timeout=timeout)
        r.raise_for_status()
        return r.json()

    def check(self):
        self.get("/system_stats")
