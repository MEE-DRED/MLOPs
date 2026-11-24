"""Streamlit dashboard for the Diabetic Retinopathy MLOps project.

Run:   streamlit run app/dashboard.py
Needs: the FastAPI backend (DR_API_URL, default http://localhost:8000). Every number shown comes from the API;
the dashboard never reads the database or model files itself. Optional DR_API_KEY is sent as X-API-Key.

NOTE: this file was written against Streamlit >= 1.37 and has been exercised in the build sandbox only with a
stand-in `streamlit` module (see tests/test_dashboard.py), NOT with real Streamlit.
"""
from __future__ import annotations

import hmac
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))     # make `src` and `app` importable

import numpy as np
import pandas as pd
import streamlit as st

from app import ui_helpers as H
from app.api_client import ApiClient, ApiError, ApiUnavailable
from src import config
from src.image_analysis import FEATURE_EXPLANATIONS

API_URL = os.environ.get("DR_API_URL", "http://localhost:8000")
FEATURES_CSV = config.TABLES_DIR / "image_features.csv"
PAGES = ["Overview", "Predict", "Dataset insights", "Bulk upload", "Retraining", "Monitoring"]
ADMIN_PAGES = {"Bulk upload", "Retraining"}      # change data/models: protected when DR_DASHBOARD_PASSWORD is set


# ------------------------------------------------------------------ plumbing
@st.cache_resource
def get_client() -> ApiClient:
    return ApiClient(API_URL, os.environ.get("DR_API_KEY", ""))


def call(fn, *args, **kwargs):
    """Call the API; show a clear error and return None on failure."""
    try:
        return fn(*args, **kwargs)
    except ApiUnavailable as exc:
        st.error("🔴 " + H.friendly_error(exc))
    except ApiError as exc:
        st.error(H.friendly_error(exc))
    return None


def disclaimer():
    st.warning("⚠️ **Experimental academic demonstration - not a medical device and not a diagnosis.** "
               "Predictions must never be used for medical decisions and cannot replace an eye examination.")


def admin_gate() -> bool:
    """True if no password is configured or the visitor has entered it. Protects pages that change data/models."""
    password = os.environ.get("DR_DASHBOARD_PASSWORD", "")
    if not password or st.session_state.get("admin_ok"):
        return True
    st.info("🔒 This page changes data or the model, so it needs the admin password.")
    entered = st.text_input("Admin password", type="password", key="admin_pw")
    if entered:
        if hmac.compare_digest(entered.encode(), password.encode()):
            st.session_state["admin_ok"] = True
            st.rerun()
        else:
            st.error("Incorrect password.")
    return False


def sidebar(client):
    """Sidebar: navigation + live API status. Returns (page, health or None)."""
    st.sidebar.title("🩺 DR Detection")
    health = None
    try:
        health = client.health()
        st.session_state["last_health_ok"] = time.time()
    except (ApiUnavailable, ApiError):
        pass
    if health is None:
        st.sidebar.error("🔴 API unreachable")
    elif health["model_loaded"]:
        st.sidebar.success(f"🟢 API online · model v{health['model_version']}")
    else:
        st.sidebar.warning("🟠 API online · NO model loaded")
    last = st.session_state.get("last_health_ok")
    st.sidebar.caption("Last successful health check: " +
                       (datetime.fromtimestamp(last).strftime("%H:%M:%S") if last else "never"))
    st.sidebar.caption(f"API: {API_URL}")
    page = st.sidebar.radio("Navigate", PAGES)
    return page, health


# ------------------------------------------------------------------ Overview
def page_overview(client, health):
    st.title("Diabetic Retinopathy Detection Using Deep Learning and MLOps")
    st.caption("Five-level severity grading of retinal fundus images with a MobileNetV2 transfer-learning model "
               "(RetinaMNIST-224), served through an API, with uploads, retraining, versioning and monitoring.")
    disclaimer()
    mon = call(client.monitoring)
    data = call(client.retraining_data)
    if mon is None or data is None:
        return
    row1 = st.columns(4)
    row1[0].metric("API status", "Online")
    row1[1].metric("Service uptime", H.uptime_text(health["uptime_seconds"]))
    row1[2].metric("Current model", f"v{health['model_version']}" if health["model_loaded"] else "none")
    row1[3].metric("Total predictions", mon["predictions"]["total"])
    events = mon["retraining"]["recent_events"]
    row2 = st.columns(4)
    row2[0].metric("Retraining images uploaded", data["n_accepted_images"])
    row2[1].metric("New since last retraining", data["n_new_images"])
    row2[2].metric("Failed prediction requests", mon["predictions"]["failed"])
    row2[3].metric("Last retraining", events[0]["status"] if events else "never run")
    if not health["model_loaded"]:
        st.error(f"No model is loaded: {health.get('model_error')}")
    st.subheader("How the system works")
    st.markdown(
        "1. **Predict** - upload one fundus image; the API validates, preprocesses and classifies it.\n"
        "2. **Dataset insights** - what the training data looks like and how the model performs.\n"
        "3. **Bulk upload** - add *correctly labelled* images (labels 0-4) for retraining.\n"
        "4. **Retraining** - fine-tune the *current* model on the uploads; the candidate is accepted only if it passes "
        "the acceptance policy, otherwise the current model keeps serving.\n"
        "5. **Monitoring** - uptime, prediction counts, uploaded data and shift indicators.")


# ------------------------------------------------------------------- Predict
def page_predict(client, health):
    st.header("Single image prediction")
    disclaimer()
    if not health["model_loaded"]:
        st.error("The API has no model loaded, so predictions are unavailable.")
        return
    upload = st.file_uploader("Upload one retinal fundus image (PNG or JPG)", type=["png", "jpg", "jpeg"])
    if upload is None:
        st.info("Choose an image to begin. Tip: the notebook exports sample images to `outputs/demo_samples/`.")
        return
    data = upload.getvalue()
    key = f"{upload.name}:{len(data)}"
    left, right = st.columns(2)
    left.image(data, caption=upload.name)
    if right.button("Classify image", type="primary"):
        with st.spinner("Running the model ..."):
            result = call(client.predict, data, upload.name)
        if result is not None:
            st.session_state["prediction"] = (key, result)
    saved = st.session_state.get("prediction")
    if not saved or saved[0] != key:
        return
    result = saved[1]
    with right:
        st.subheader(f"{result['predicted_class']} - {result['predicted_class_name']}")
        st.metric("Confidence", f"{100 * result['confidence']:.1f}%")
        st.caption(f"Model version {result['model_version']}")
        st.bar_chart(H.probabilities_frame(result))
    st.caption("Probabilities are the model's softmax scores over the five grades. They are not calibrated "
               "medical risk estimates, and neighbouring grades are often confused (see Dataset insights).")
    st.warning(result.get("disclaimer", config.DISCLAIMER))


# ----------------------------------------------------------- Dataset insights
@st.cache_data(ttl=3600)
def load_features() -> pd.DataFrame | None:
    return pd.read_csv(FEATURES_CSV) if FEATURES_CSV.is_file() else None


def page_insights(client, health):
    import matplotlib.pyplot as plt
    from src import visualization as viz

    st.header("Dataset insights")
    models = call(client.models)
    if models is None:
        return
    tab_classes, tab_images, tab_perf = st.tabs(["Class distribution", "Image characteristics", "Model performance"])

    with tab_classes:
        counts = H.dataset_counts(models)
        if counts is None:
            st.info("No dataset information is recorded in the model metadata yet.")
        else:
            df = pd.DataFrame(counts, index=list(config.CLASS_NAMES))
            st.subheader("Images per DR grade in each split")
            st.bar_chart(df)
            pct = (df / df.sum() * 100).round(1)
            st.dataframe(pct.rename(columns=lambda c: f"{c} %"))
            st.caption("The classes are imbalanced: 'No DR' is about 45% of the training set and 'Proliferative DR' "
                       "about 6%. Training therefore uses class weights, and macro-F1 (not accuracy) is the selection "
                       "metric. The test set has only 20 Proliferative images, so that class's scores are noisy.")
        data = call(client.retraining_data)
        if data and data["n_accepted_images"]:
            st.subheader("Images uploaded for retraining")
            st.bar_chart(H.class_count_frame(data["class_counts"]))

    with tab_images:
        feats = load_features()
        if feats is None:
            st.info(f"Image-characteristics table not found at `{FEATURES_CSV}`. Run the notebook (section 9b) "
                    "and copy `image_features.csv` into `outputs/tables/`.")
        else:
            st.subheader("Brightness, contrast and sharpness")
            for name in ("brightness", "contrast", "sharpness"):
                st.markdown(f"**{name.capitalize()}** - {FEATURE_EXPLANATIONS[name]}")
            fig = viz.plot_feature_distributions(feats)
            st.pyplot(fig)
            plt.close(fig)
            summ = H.feature_summaries(feats)
            st.markdown("**Median per DR grade**")
            st.dataframe(summ["medians"])
            st.markdown("**Rank correlation with DR grade** (computed from the table above)")
            st.dataframe(summ["rho"])
            st.markdown("**Share of images with the narrower retinal framing** (retina area about 66% vs about 72% of the image)")
            st.dataframe(summ["framing"])
            st.info("What this tells us: brightness is clearly higher in 'No DR' images, contrast is almost the same in every "
                    "grade, sharpness mostly reflects focus quality, and the camera framing changes with grade. These are "
                    "properties of the *photographs*, not causes or signs of disease. They are a warning that a model could "
                    "partly learn camera or exposure differences instead of lesions - which is why the notebook compares the "
                    "CNN with a baseline that sees only these statistics.")

    with tab_perf:
        view = H.performance_view(models)
        if view is None or not view["metrics"]:
            st.info("No evaluated model is available yet.")
        else:
            m = view["metrics"]
            st.subheader(f"Model v{view['version']} - {view['split']} set (n={m['n_samples']})")
            if view["note"]:
                st.caption(view["note"])
            top = st.columns(4)
            top[0].metric("Accuracy", f"{m['accuracy']:.3f}")
            top[1].metric("Macro F1", f"{m['f1_macro']:.3f}")
            top[2].metric("Quadratic weighted kappa", f"{m['quadratic_weighted_kappa']:.3f}")
            top[3].metric("Majority-class accuracy", f"{m['majority_class_baseline_accuracy']:.3f}")
            st.dataframe(H.headline_table(m))
            st.markdown("**Per-class precision, recall and F1**")
            st.dataframe(H.per_class_table(m))
            cm = np.array(m["confusion_matrix"])
            c1, c2 = st.columns(2)
            with c1:
                fig = viz.plot_confusion_matrix(cm, title="Counts")
                st.pyplot(fig)
                plt.close(fig)
            with c2:
                fig = viz.plot_confusion_matrix(cm, normalize=True, title="Row-normalised (recall)")
                st.pyplot(fig)
                plt.close(fig)
            st.caption("Most errors are between neighbouring severity grades, which are subtle even for specialists. "
                       "Accuracy should be read next to the majority-class accuracy: always answering 'No DR' already scores that high.")
        st.markdown("**All model versions**")
        st.dataframe(H.versions_table(models))
        st.caption("Grad-CAM visualisations (model attention, not lesion detection) are in the notebook, section 23.")


# --------------------------------------------------------------- Bulk upload
def page_upload(client, health):
    st.header("Bulk upload of labelled images for retraining")
    st.info("Every image needs its **correct** DR grade (0-4). Images without a valid label are never stored. "
            "Byte-identical duplicates, unreadable files and images that are identical to the original "
            "validation/test images are rejected.")
    files = st.file_uploader("Select retinal images (PNG or JPG)", type=["png", "jpg", "jpeg"],
                             accept_multiple_files=True)
    summary = call(client.retraining_data)
    if summary:
        cols = st.columns(3)
        cols[0].metric("Valid labelled images stored", summary["n_accepted_images"])
        cols[1].metric("Invalid uploads so far", summary["invalid_uploads"])
        cols[2].metric("Classes covered", summary["n_classes"])
    if not files:
        return
    choices = [H.LABEL_PLACEHOLDER] + H.label_choices()
    keys = [f"label_{i}_{f.name}_{f.size}" for i, f in enumerate(files)]

    def set_all():
        for k in keys:
            st.session_state[k] = st.session_state["set_all_choice"]

    st.selectbox("Give every image the same label (optional shortcut)", choices, key="set_all_choice")
    st.button("Apply to all images", on_click=set_all)
    st.markdown("**Assign the correct label to each image**")
    labels = []
    for f, k in zip(files, keys):
        c_img, c_name, c_sel = st.columns([1, 3, 3])
        c_img.image(f.getvalue(), width=80)
        c_name.write(f.name)
        choice = c_sel.selectbox("Label", choices, key=k, label_visibility="collapsed")
        labels.append(H.choice_to_label(choice))
    missing = sum(lab is None for lab in labels)
    if missing:
        st.warning(f"{missing} of {len(files)} image(s) still need a label.")
    if st.button(f"Upload {len(files)} labelled image(s)", type="primary", disabled=missing > 0):
        with st.spinner("Uploading and validating ..."):
            report = call(client.upload_retraining_data, [(f.name, f.getvalue()) for f in files], labels)
        if report is not None:
            st.session_state["upload_report"] = report
    report = st.session_state.get("upload_report")
    if report:
        st.subheader("Upload result")
        a, b, c = st.columns(3)
        a.metric("Accepted", report["n_accepted"])
        b.metric("Rejected", report["n_rejected"])
        c.metric("Total stored images", report["total_available_images"])
        st.dataframe(H.upload_results_table(report))
        if report["n_rejected"]:
            st.error("Some files were rejected - see the 'reason' column. Rejected files were not stored.")
        else:
            st.success("All files were accepted.")
        if report.get("auto_retraining_started"):
            st.info("Automatic retraining was triggered by this upload - follow it on the Retraining page.")
        st.caption(f"Data-leakage check against the original dataset: {report['leakage_check']}.")


# ---------------------------------------------------------------- Retraining
def _elapsed(started_iso: str) -> str:
    started = datetime.fromisoformat(started_iso)
    return H.uptime_text((datetime.now(timezone.utc) - started).total_seconds())


def page_retrain(client, health):
    st.header("Retraining")
    st.markdown("Retraining **fine-tunes the current model** on the uploaded labelled images (mixed with a sample of the "
                "original training data). The candidate and the current model are both scored on the original "
                "validation set. The candidate replaces the current model **only** if it passes the acceptance policy; "
                "otherwise it is kept as a rejected version and nothing changes for users. The test set is never used.")
    data = call(client.retraining_data)
    status = call(client.retrain_status)
    if data is None or status is None:
        return
    running = bool(status["running"])
    if not running:
        st.session_state["retrain_clicked"] = False

    cols = st.columns(4)
    cols[0].metric("Valid labelled images", data["n_accepted_images"])
    cols[1].metric("New since last attempt", data["n_new_images"])
    cols[2].metric("Classes covered", f"{data['n_classes']} (min {data['min_classes']})")
    cols[3].metric("Images required", data["min_images"])
    if data["n_accepted_images"]:
        st.bar_chart(H.class_count_frame(data["class_counts"]))
    (st.success if data["eligible"] else st.warning)(data["message"])

    clicked = st.button("🔁 Retrain model now", type="primary",
                        disabled=running or not data["eligible"] or st.session_state.get("retrain_clicked", False))
    started = False
    if clicked:
        st.session_state["retrain_clicked"] = True          # blocks a second click during the rerun
        try:
            result = client.retrain()
            st.session_state["was_running"] = True
            started = True
        except ApiError as exc:
            st.session_state["retrain_clicked"] = False
            (st.warning if exc.status == 409 else st.error)(H.friendly_error(exc))
        except ApiUnavailable as exc:
            st.session_state["retrain_clicked"] = False
            st.error(H.friendly_error(exc))
    if started:
        st.rerun()

    @st.fragment(run_every=3 if running else None)
    def status_panel():
        s = call(client.retrain_status)
        if s is None:
            return
        now_running = bool(s["running"])
        if st.session_state.get("was_running") and not now_running:
            st.session_state["was_running"] = False
            st.session_state["retrain_clicked"] = False
            st.rerun()                                      # refresh metrics, button and model version
        st.session_state["was_running"] = now_running
        ev = s["event"]
        st.subheader("Latest retraining job")
        if ev is None:
            st.info("No retraining has been run yet.")
            return
        st.markdown(f"**Job #{ev['id']}** ({ev['trigger']}) - {H.status_badge(ev['status'])}")
        if ev["status"] == "running":
            st.info(f"⏳ Training in progress for {_elapsed(ev['started_at'])}. The API does not report per-epoch "
                    "progress; this panel checks the job status every 3 seconds. Please keep this page open.")
            return
        st.write(ev["message"])
        table = H.metrics_comparison(ev)
        if table is not None:
            st.markdown("**Validation metrics (original validation split)**")
            st.dataframe(table)
        acceptance = ev.get("acceptance") or {}
        if acceptance.get("reasons"):
            st.markdown("**Acceptance decision:** " + ("accepted" if acceptance.get("accepted") else "rejected"))
            for reason in acceptance["reasons"]:
                st.write(f"- {reason}")
        if ev.get("candidate_version"):
            st.caption(f"Candidate registered as version {ev['candidate_version']}.")

    status_panel()

    models = call(client.models)
    if models:
        st.subheader("Model versions")
        st.caption(f"Currently serving version {models['current_version']}.")
        st.dataframe(H.versions_table(models))
        with st.expander("Roll back to the previous version"):
            st.write("Switches the prediction service back to the previously served version. No files are deleted.")
            sure = st.checkbox("I understand this changes the model used for predictions")
            if st.button("Roll back", disabled=not sure or running):
                res = call(client.rollback)
                if res is not None:
                    st.success(f"Now serving version {res['rolled_back_to']}.")
                    st.rerun()


# ---------------------------------------------------------------- Monitoring
def page_monitoring(client, health):
    st.header("Monitoring")
    auto = st.checkbox("Auto-refresh every 5 seconds", value=False)

    @st.fragment(run_every=5 if auto else None)
    def panel():
        mon = call(client.monitoring)
        if mon is None:
            return
        svc, pred, data = mon["service"], mon["predictions"], mon["retraining_data"]

        st.subheader("Service and model uptime")
        a = st.columns(4)
        a[0].metric("API status", "Online" if svc["status"] in ("ok", "degraded") else svc["status"])
        a[1].metric("Model available", "Yes" if svc["model_loaded"] else "No")
        a[2].metric("Service uptime", svc["uptime"])
        a[3].metric("Model version", f"v{svc['model_version']}" if svc["model_loaded"] else "none")
        last = st.session_state.get("last_health_ok")
        st.caption("Last successful health check: " + (datetime.fromtimestamp(last).strftime("%H:%M:%S") if last else "never")
                   + ". Uptime restarts when the service restarts.")

        st.subheader("Predictions")
        b = st.columns(4)
        b[0].metric("Total requests", pred["total"])
        b[1].metric("Successful", pred["successful"])
        b[2].metric("Failed", pred["failed"])
        b[3].metric("Avg latency (ms)", "-" if pred["avg_latency_ms"] is None else f"{pred['avg_latency_ms']:.0f}")
        if pred["successful"]:
            st.markdown("**Predicted class distribution**")
            st.bar_chart(H.class_count_frame(pred["by_class"]))
        if pred["errors"]:
            st.write("Failure reasons:", pred["errors"])
        if pred["recent"]:
            st.markdown("**Recent predictions**")
            st.dataframe(pd.DataFrame(pred["recent"]))

        st.subheader("Retraining data")
        c = st.columns(4)
        c[0].metric("Stored images", data["available_images"])
        c[1].metric("New since last retraining", data["new_since_last_retraining"])
        c[2].metric("Invalid uploads", data["invalid_uploads"])
        c[3].metric("Excluded at retraining", data["excluded_uploads"])
        if data["available_images"]:
            st.bar_chart(H.class_count_frame(data["class_counts"]))
        if data["recent_uploads"]:
            st.markdown("**Recent uploads**")
            st.dataframe(pd.DataFrame(data["recent_uploads"]))

        st.subheader("Shift indicators")
        st.caption("Two different things are shown. They are indicators, not proof that the model has degraded.")
        drift = mon["drift"]
        cd = drift.get("class_distribution")
        if cd:
            st.markdown("**1. Class-distribution change (labels only - this is NOT image drift)**")
            if cd["assessable"]:
                st.metric("Total variation distance vs training labels", f"{cd['total_variation_distance']:.3f}")
                st.dataframe(pd.DataFrame({"uploaded share": cd["observed_share"], "training share": cd["reference_share"]},
                                          index=list(config.CLASS_NAMES)))
            else:
                st.caption(cd.get("note", "Not enough data."))
        im = drift.get("image_statistics")
        if im:
            st.markdown("**2. Image-statistics shift (KS test on brightness, contrast, sharpness, retina area)**")
            if im["assessable"]:
                st.dataframe(pd.DataFrame(im["features"]).T)
                (st.warning if im.get("any_flagged") else st.success)(
                    "At least one statistic differs from the training images." if im.get("any_flagged")
                    else "No statistic differs significantly from the training images.")
                st.caption(im["note"])
            else:
                st.caption(im.get("note", "Not enough data."))
        if drift.get("error"):
            st.caption(f"Shift indicators unavailable: {drift['error']}")
        if not cd and not im:
            st.caption("Shift indicators need the original dataset on the API server.")

        st.subheader("Retraining events")
        r = mon["retraining"]
        st.caption(f"Automatic retraining: {'ON' if r['auto_retraining_enabled'] else 'off'} "
                   f"(threshold: {r['auto_threshold_new_images']} new valid images). "
                   f"A job is currently {'running' if r['running'] else 'not running'}.")
        if r["recent_events"]:
            st.dataframe(H.events_table(r["recent_events"]))
        else:
            st.caption("No retraining events yet.")
        if mon["model_versions"]:
            st.subheader("Registered model versions")
            st.dataframe(pd.DataFrame(mon["model_versions"]))

    panel()


# ---------------------------------------------------------------------- main
PAGE_FUNCS = {"Overview": page_overview, "Predict": page_predict, "Dataset insights": page_insights,
              "Bulk upload": page_upload, "Retraining": page_retrain, "Monitoring": page_monitoring}


def main():
    st.set_page_config(page_title="DR Detection - MLOps", page_icon="🩺", layout="wide")
    client = get_client()
    page, health = sidebar(client)
    if health is None:
        st.title("Diabetic Retinopathy Detection")
        st.error(f"🔴 The API at {API_URL} is not reachable, so nothing can be shown. Start it with "
                 "`uvicorn src.api:app --port 8000` (or `docker compose up`) and reload this page.")
        st.stop()
        return
    if page in ADMIN_PAGES and not admin_gate():
        return
    PAGE_FUNCS[page](client, health)


if __name__ == "__main__":      # Streamlit executes the script as __main__
    main()
