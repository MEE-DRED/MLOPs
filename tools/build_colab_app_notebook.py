"""Generates notebook/colab_run_app_and_tests.ipynb (real-stack verification in Colab)."""
import json
from pathlib import Path

cells = []
md = lambda t: cells.append({"cell_type": "markdown", "metadata": {}, "source": t.strip("\n").splitlines(True)})
code = lambda t: cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                               "source": t.strip("\n").splitlines(True)})

md('''
# Run and verify the full application in Colab (real TensorFlow + FastAPI + Streamlit)
This notebook is the **first real-stack test** of the project: the unit tests, the live API with your actual trained model,
the end-to-end check, a flood-request simulation and the real Streamlit dashboard. Nothing here is assumed to work - read the outputs.

**Prerequisites (in Google Drive, folder `MyDrive/DR_MLOps/`):**
1. `Diabetic-Retinopathy-MLOps.zip` - the current project zip.
2. `models/` - written by the training notebook (`model_v*.keras`, `metadata/`, `current_model.json`).
3. `data/retinamnist_224.npz` - the dataset (needed for retraining + shift indicators; predictions work without it).
4. `outputs/demo_samples/` - created by training-notebook section 26 (correctly classified test images, used by the e2e check).

Runtime: CPU is fine (no training happens here).
''')
code('''
import os, sys, shutil, subprocess, zipfile, json, time
from pathlib import Path
from google.colab import drive

drive.mount("/content/drive")
DRIVE = Path("/content/drive/MyDrive/DR_MLOps")
ZIP, CODE = DRIVE / "Diabetic-Retinopathy-MLOps.zip", Path("/content/dr_code")
assert ZIP.exists(), f"Put the project zip at {ZIP}"
shutil.rmtree(CODE, ignore_errors=True)
zipfile.ZipFile(ZIP).extractall(CODE)
REPO = next(p.parent.parent for p in CODE.rglob("src/config.py"))
os.chdir(REPO)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "fastapi", "uvicorn[standard]", "python-multipart", "httpx", "requests", "streamlit"], check=True)
import tensorflow as tf, keras, fastapi, streamlit
print("tensorflow", tf.__version__, "| keras", keras.__version__, "| fastapi", fastapi.__version__, "| streamlit", streamlit.__version__)
''')
code('''
STATE = Path("/content/dr_state"); shutil.rmtree(STATE, ignore_errors=True)
ENV = {**os.environ, "PYTHONPATH": str(REPO), "DR_STATE_DIR": str(STATE), "DR_BAKED_MODELS_DIR": str(DRIVE / "models"),
       "DR_MODELS_DIR": str(STATE / "models"), "DR_DB_PATH": str(STATE / "database" / "dr_mlops.sqlite3"),
       "DR_UPLOAD_DIR": str(STATE / "uploads"), "DR_DATASET_NPZ": str(DRIVE / "data" / "retinamnist_224.npz"),
       "DR_API_URL": "http://127.0.0.1:8000", "TF_CPP_MIN_LOG_LEVEL": "2",
       "DR_TEST_NPZ": str(DRIVE / "data" / "retinamnist_224.npz")}      # lets the real-dataset unit tests run too
os.environ.update(ENV)
r = subprocess.run([sys.executable, "-m", "scripts.prepare_runtime"], env=ENV, capture_output=True, text=True)
print(r.stdout, r.stderr[-1500:])
''')
md('''
## 1. Unit tests with the real dependencies
Locally in the build sandbox 6 API tests are skipped because FastAPI is missing. Here FastAPI and httpx are installed, so **they run**.
Look for `skipped=` in the summary: it should now be 0 (or only tests needing something else).
''')
code('''
r = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-t", ".", "-v"], env=ENV, capture_output=True, text=True)
lines = r.stderr.strip().splitlines()
print("\\n".join(l for l in lines if "ApiTests" in l))          # the 6 HTTP-layer tests
print("\\n".join(lines[-6:]))
''')
md('## 2. Start the API with your real trained model')
code('''
api_log = open("/content/api.log", "w")
api = subprocess.Popen([sys.executable, "-m", "uvicorn", "src.api:app", "--host", "127.0.0.1", "--port", "8000"],
                       env=ENV, stdout=api_log, stderr=subprocess.STDOUT)
import urllib.request
health = None
for _ in range(90):
    try:
        health = json.loads(urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=3).read()); break
    except Exception:
        if api.poll() is not None: break
        time.sleep(2)
print(json.dumps(health, indent=2) if health else open("/content/api.log").read()[-3000:])
assert health and health["model_loaded"], "API did not load the model - read the log above"
''')
md('''
## 3. End-to-end check against the live API
Exercises health, real predictions on your demo images, invalid-file rejection, monitoring counters, invalid-label rejection,
server-side validation, the retraining-eligibility message and the model registry. Exit status = every check passed.
''')
code('''
r = subprocess.run([sys.executable, "-m", "scripts.e2e_check", "--url", "http://127.0.0.1:8000",
                    "--samples", str(DRIVE / "outputs" / "demo_samples"), "--out", str(DRIVE / "outputs" / "reports" / "e2e_check.json")],
                   env=ENV, capture_output=True, text=True)
print(r.stdout or r.stderr[-2000:]); print("exit code:", r.returncode)
''')
md('''
## 4. Flood-request simulation (for the README)
Fires concurrent requests at `/predict` and `/health`. These numbers describe **this Colab VM**, not the deployed Space - repeat
`python -m scripts.flood_test --url <your-deployed-api-or-space-internal-url>` where it matters, and paste only measured values into the README.
''')
code('''
sample = sorted((DRIVE / "outputs" / "demo_samples").glob("true*"))[0]
for args in (["--endpoint", "predict", "--requests", "100", "--concurrency", "5", "--image", str(sample)],
             ["--endpoint", "health", "--requests", "500", "--concurrency", "50"]):
    r = subprocess.run([sys.executable, "-m", "scripts.flood_test", "--url", "http://127.0.0.1:8000",
                        "--out-dir", str(DRIVE / "outputs" / "reports"), *args], env=ENV, capture_output=True, text=True)
    print(r.stdout[-900:] or r.stderr[-1500:]); print("-" * 60)
''')
md('''
## 5. Real Streamlit: every dashboard page
Uses Streamlit's own headless test runner (`AppTest`) against the live API. **Any exception raised while a page renders is shown.**
(File uploads and button clicks are exercised by hand in step 6 - this step proves every page renders with real data.)
''')
code('''
from streamlit.testing.v1 import AppTest
failures = {}
for page in ["Overview", "Predict", "Dataset insights", "Bulk upload", "Retraining", "Monitoring"]:
    at = AppTest.from_file("app/dashboard.py", default_timeout=90).run()
    at.sidebar.radio[0].set_value(page).run()
    errs = [str(e.value) for e in at.exception]
    failures[page] = errs
    print(f"{page:<18} exceptions: {len(errs)}   metrics shown: {len(at.metric)}   errors shown: {[e.value[:80] for e in at.error]}")
    for e in errs: print("    ", e[:400])
assert not any(failures.values()), "At least one page raised an exception - see above"
''')
md('''
## 6. Use the dashboard (for your video / manual testing)
Opens the real dashboard inside Colab. Try: **Predict** (upload a demo sample), **Dataset insights**, **Monitoring**.
For **Bulk upload / Retraining** you need genuinely new, correctly labelled fundus images (re-uploading dataset images is rejected by design).
''')
code('''
ui_log = open("/content/ui.log", "w")
ui = subprocess.Popen([sys.executable, "-m", "streamlit", "run", "app/dashboard.py", "--server.port", "8501", "--server.address", "127.0.0.1",
                       "--server.headless", "true", "--browser.gatherUsageStats", "false"], env=ENV, stdout=ui_log, stderr=subprocess.STDOUT)
for _ in range(40):
    try:
        urllib.request.urlopen("http://127.0.0.1:8501/_stcore/health", timeout=3); break
    except Exception: time.sleep(2)
from google.colab import output
output.serve_kernel_port_as_iframe(8501, height=950)
''')
md('''
*Want a public URL for a quick demo?* Colab can tunnel the port, e.g. with Cloudflare's quick tunnel (no account; the URL dies with the runtime).
This is optional, unverified here and **not** the project's deployment - the deployment target is the Hugging Face Space (see `docs/deployment.md`).
''')
code('''
# OPTIONAL quick public URL (uncomment to use)
# subprocess.run("wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -O /content/cloudflared && chmod +x /content/cloudflared", shell=True)
# tunnel = subprocess.Popen(["/content/cloudflared", "tunnel", "--url", "http://127.0.0.1:8501"], stderr=subprocess.PIPE, text=True)
# for line in tunnel.stderr:
#     if "trycloudflare.com" in line: print(line.strip()); break
''')
md('## 7. Stop the servers')
code('''
for p in (globals().get("ui"), globals().get("api")):
    if p and p.poll() is None: p.terminate()
print("stopped")
''')

md('''
## 8. (Optional) Deploy to a Hugging Face Docker Space - straight from Colab
Nothing is downloaded to your laptop. **You must do these manual steps first** (they need your account; I cannot do them):
1. Create a free account at https://huggingface.co and verify your e-mail.
2. Settings -> Access Tokens -> create a token with **Write** permission.
3. In Colab: the key icon (Secrets) -> add a secret named `HF_TOKEN` with that token and enable notebook access. *Never paste the token into a cell.*

The bundle contains only what the container needs (code, models, dataset, Space README). A Space is **public unless you choose Private**; the dataset and
model are then visible to anyone who can see the Space.
''')
code('''
HF_USERNAME = "your-hf-username"          # <-- edit
SPACE_NAME = "dr-mlops"
BUNDLE = Path("/content/space_bundle")
from google.colab import userdata
HF_TOKEN = userdata.get("HF_TOKEN")        # raises if the secret is missing

# models must be in the repo's models/ folder for the bundle (copy them from Drive; the pointer file goes last)
shutil.copytree(DRIVE / "models", REPO / "models", dirs_exist_ok=True)
shutil.rmtree(BUNDLE, ignore_errors=True)
r = subprocess.run([sys.executable, "-m", "scripts.prepare_space_bundle", "--out", str(BUNDLE),
                    "--dataset", str(DRIVE / "data" / "retinamnist_224.npz")], env=ENV, capture_output=True, text=True)
print(r.stdout or r.stderr)
''')
code('''
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "huggingface_hub"], check=True)
from huggingface_hub import HfApi
api_hf = HfApi(token=HF_TOKEN)
repo_id = f"{HF_USERNAME}/{SPACE_NAME}"
api_hf.create_repo(repo_id=repo_id, repo_type="space", space_sdk="docker", exist_ok=True, private=False)
api_hf.upload_folder(folder_path=str(BUNDLE), repo_id=repo_id, repo_type="space", commit_message="Deploy DR MLOps demo")
print("Pushed. Build logs and status:", f"https://huggingface.co/spaces/{repo_id}")
print("The first build takes several minutes. It is NOT deployed until the Space shows 'Running' - then follow docs/deployment.md section 'Verification checklist'.")
''')

md('''
## 9. Build the final submission ZIP (rubric attempt 1)
Copies your trained `models/` from Drive into the repository folder and zips the repository. **The builder refuses to run without a trained model**, excludes the dataset / databases / caches / `.env`,
and aborts if it finds credential-looking text. Read the manifest it prints. See `docs/submission_checklist.md` for everything else (video link, GitHub repo).
''')
code('''
shutil.copytree(DRIVE / "models", REPO / "models", dirs_exist_ok=True)
SUB_ZIP = DRIVE / "Diabetic-Retinopathy-MLOps_submission.zip"
r = subprocess.run([sys.executable, "-m", "scripts.make_submission_zip", "--out", str(SUB_ZIP)], env=ENV, capture_output=True, text=True)
print(r.stdout or r.stderr)
''')

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python"}, "colab": {"provenance": []}}, "nbformat": 4, "nbformat_minor": 4}
out = Path(__file__).resolve().parents[1] / "notebook" / "colab_run_app_and_tests.ipynb"
out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print("wrote", out, "-", len(cells), "cells")
