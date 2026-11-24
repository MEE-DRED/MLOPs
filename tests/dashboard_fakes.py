"""Test doubles for the dashboard: a stand-in `streamlit` module and an in-process HTTP transport.

LocalTransport routes ApiClient requests straight into AppService and maps exceptions to status codes exactly
as src/api.py does. It exercises dashboard -> client -> service -> database/registry. It does NOT test FastAPI
request parsing or real Streamlit rendering (neither is installed in the build sandbox).
"""
import types

from src.model_registry import RegistryError
from src.prediction import ModelNotLoadedError
from src.preprocessing import InvalidImageError
from src.retrain import RetrainingBusy, RetrainingNotPossible


# --------------------------------------------------------------- transport
class Resp:
    def __init__(self, status, payload):
        self.status_code, self._payload = status, payload
        self.text = str(payload)

    def json(self):
        return self._payload


class LocalTransport:
    def __init__(self, svc, api_key=""):
        self.svc, self.api_key, self.down, self.log = svc, api_key, False, []

    def request(self, method, url, headers=None, timeout=None, params=None, files=None, data=None):
        if self.down:
            raise ConnectionError("connection refused")
        path = "/" + url.split("//", 1)[1].split("/", 1)[1]
        self.log.append((method, path))
        mutating = path in ("/upload-retraining-data", "/retrain", "/models/rollback")
        if mutating and self.api_key and (headers or {}).get("X-API-Key") != self.api_key:
            return Resp(401, {"detail": "Missing or invalid X-API-Key header"})
        s = self.svc
        try:
            if (method, path) == ("GET", "/health"): return Resp(200, s.health())
            if (method, path) == ("POST", "/predict"): return Resp(200, s.predict(files["file"][1]))
            if (method, path) == ("POST", "/upload-retraining-data"):
                return Resp(200, s.upload_retraining_data([(n, b) for _, (n, b, _m) in files], data["labels"]))
            if (method, path) == ("GET", "/retraining-data"): return Resp(200, s.retraining_data_summary())
            if (method, path) == ("POST", "/retrain"):
                return Resp(202, s.retrain(wait=(params or {}).get("wait") == "true"))
            if (method, path) == ("GET", "/retrain/status"):
                return Resp(200, s.retrain_status((params or {}).get("event_id")))
            if (method, path) == ("GET", "/monitoring"): return Resp(200, s.monitoring())
            if (method, path) == ("GET", "/models"): return Resp(200, s.models())
            if (method, path) == ("POST", "/models/rollback"): return Resp(200, s.rollback())
            return Resp(404, {"detail": "not found"})
        except InvalidImageError as exc: return Resp(422, {"detail": f"Invalid image: {exc}"})
        except ModelNotLoadedError as exc: return Resp(503, {"detail": str(exc)})
        except ValueError as exc: return Resp(422, {"detail": str(exc)})
        except RetrainingNotPossible as exc: return Resp(422, {"detail": {"message": str(exc), **exc.detail}})
        except (RetrainingBusy, RegistryError) as exc: return Resp(409, {"detail": str(exc)})


# --------------------------------------------------------------- fake streamlit
class StopApp(Exception):
    pass


class SessionState(dict):
    def __getattr__(self, k): return self.get(k)
    def __setattr__(self, k, v): self[k] = v


class FakeUpload:
    def __init__(self, name, data):
        self.name, self._d, self.size = name, data, len(data)

    def getvalue(self): return self._d


class Ctx:
    """Context manager that forwards any attribute (metric, image, ...) to the fake module."""

    def __init__(self, st): self._st = st
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def __getattr__(self, name): return getattr(self._st, name)


class FakeStreamlit(types.ModuleType):
    def __init__(self):
        super().__init__("streamlit")
        self.session_state = SessionState()
        self.calls = []                      # (name, args, kwargs)
        self.click, self.uploads, self.page = set(), None, "Overview"
        self.checkbox_values, self.selectbox_values = {}, {}
        self.button_states, self.fragments = [], []
        self.text_inputs = {}
        self.sidebar = Ctx(self)

    # recording helper -------------------------------------------------
    def of(self, name): return [c for c in self.calls if c[0] == name]
    def texts(self, name): return [str(c[1][0]) for c in self.of(name) if c[1]]

    def _rec(self, name, *a, **k):
        self.calls.append((name, a, k))

    # display elements (all just recorded) ------------------------------
    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        if name in ("title", "header", "subheader", "caption", "markdown", "write", "info", "warning", "error",
                    "success", "metric", "dataframe", "bar_chart", "image", "pyplot", "set_page_config"):
            return lambda *a, **k: self._rec(name, *a, **k)
        raise AttributeError(name)

    def spinner(self, *a, **k): return Ctx(self)
    def expander(self, *a, **k): return Ctx(self)
    def columns(self, spec, **k): return [Ctx(self) for _ in range(spec if isinstance(spec, int) else len(spec))]
    def tabs(self, labels): return [Ctx(self) for _ in labels]
    def stop(self): self._rec("stop"); raise StopApp()
    def rerun(self, **k): self._rec("rerun")

    def cache_resource(self, *a, **k):
        return a[0] if a and callable(a[0]) else (lambda f: f)
    cache_data = cache_resource

    def fragment(self, *a, run_every=None, **k):
        self.fragments.append(run_every)
        return a[0] if a and callable(a[0]) else (lambda f: f)

    # widgets -------------------------------------------------------------
    def radio(self, label, options, **k): return self.page
    def checkbox(self, label, value=False, **k): return self.checkbox_values.get(label, value)

    def selectbox(self, label, options, key=None, **k):
        for prefix, val in self.selectbox_values.items():
            if key and key.startswith(prefix):
                self.session_state[key] = val
        value = self.session_state.get(key, options[0]) if key else options[0]
        if key: self.session_state[key] = value
        return value

    def file_uploader(self, label, **k): return self.uploads

    def text_input(self, label, key=None, **k):
        self._rec('text_input', label)
        return self.text_inputs.get(key, '')

    def button(self, label, type=None, disabled=False, on_click=None, **k):
        self.button_states.append((label, bool(disabled)))
        if disabled:
            return False
        hit = next((c for c in self.click if c in label), None)
        if hit is None:
            return False
        self.click.discard(hit)
        if on_click: on_click()
        return True
