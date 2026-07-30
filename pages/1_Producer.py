"""
Producer page - simulates the live MES/quality API feed. Click "Push next
batch" to release the next few rows from the held-out log pool into the
shared live queue, as if new production had just come off the line.

Aligned to AI4I 2020 Predictive Maintenance Dataset standard.
"""

import os
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Producer - simulate live feed", layout="wide")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
POOL_PATH = os.path.join(DATA_DIR, "log_pool.csv")
QUEUE_PATH = os.path.join(DATA_DIR, "live_queue.csv")

st.title("🔄 Producer — Simulate Live MES Feed")
st.caption(
    "This page plays the role of the MES + quality inspection APIs in a real "
    "deployment. Pushing a batch here is equivalent to a new batch arriving "
    "live in production. The model has **never seen** these batches."
)


@st.cache_data(show_spinner=False)
def load_pool():
    return pd.read_csv(POOL_PATH)


pool_df = load_pool()

# These columns are the "answer key" for demo validation only — a real MES
# feed would never include them, so they're stripped before pushing
HIDDEN_COLS = {"ground_truth_label", "failure_modes", "Machine failure", "TWF", "HDF", "PWF", "OSF", "RNF"}
COLUMNS_TO_PUSH = [c for c in pool_df.columns if c not in HIDDEN_COLS]

if "pushed_count" not in st.session_state:
    if os.path.exists(QUEUE_PATH):
        pushed_ids = set(pd.read_csv(QUEUE_PATH)["Product ID"])
        st.session_state.pushed_count = int(pool_df["Product ID"].isin(pushed_ids).sum())
    else:
        st.session_state.pushed_count = 0

remaining = len(pool_df) - st.session_state.pushed_count

# --- KPIs ---
col1, col2, col3 = st.columns(3)
col1.metric("📤 Batches pushed", st.session_state.pushed_count)
col2.metric("📦 Remaining in pool", remaining)

# Show ground truth stats for the pool
n_failures = pool_df["ground_truth_label"].value_counts().get("FAILURE", 0)
col3.metric("⚠️ Failures in pool", f"{n_failures} / {len(pool_df)}")

st.markdown("---")

batch_size = st.slider("Batch size to push", 1, 20, 5)

push_col, reset_col = st.columns(2)

with push_col:
    if st.button("🚀 Push next batch", type="primary", disabled=remaining <= 0):
        start = st.session_state.pushed_count
        end = min(start + batch_size, len(pool_df))
        new_rows = pool_df.iloc[start:end][COLUMNS_TO_PUSH]

        header = not os.path.exists(QUEUE_PATH)
        new_rows.to_csv(QUEUE_PATH, mode="a", header=header, index=False)

        st.session_state.pushed_count = end

        # Show what was pushed and whether any were failures
        pushed_slice = pool_df.iloc[start:end]
        n_fail = (pushed_slice["ground_truth_label"] == "FAILURE").sum()
        if n_fail > 0:
            st.success(
                f"Pushed {len(new_rows)} batch(es) — **{n_fail} contain failures** "
                f"(the model should catch them). Switch to Dashboard to see results."
            )
        else:
            st.success(
                f"Pushed {len(new_rows)} batch(es) — all normal. "
                f"Switch to Dashboard to see them scored."
            )

with reset_col:
    if st.button("🔄 Reset simulation"):
        for fpath in [QUEUE_PATH,
                      os.path.join(DATA_DIR, ".processed_batch_ids.txt"),
                      os.path.join(DATA_DIR, "alerts.csv"),
                      os.path.join(DATA_DIR, "feedback.csv")]:
            if os.path.exists(fpath):
                os.remove(fpath)
        st.session_state.pushed_count = 0
        st.cache_data.clear()
        st.rerun()

# --- Preview ---
st.markdown("---")
st.subheader("👀 Preview: Next batches to be pushed")

start = st.session_state.pushed_count
preview = pool_df.iloc[start:start + batch_size]

if len(preview):
    # Show with ground truth so the user can see what's coming
    highlight_cols = ["Product ID", "Type", "Air temperature [K]",
                      "Process temperature [K]", "Rotational speed [rpm]",
                      "Torque [Nm]", "Tool wear [min]",
                      "ground_truth_label", "failure_modes"]
    available = [c for c in highlight_cols if c in preview.columns]
    st.dataframe(preview[available], use_container_width=True)
else:
    st.info("All batches have been pushed! Hit Reset to start over.")

# --- Pool statistics ---
with st.expander("📊 Full pool statistics"):
    st.markdown(f"**Total rows:** {len(pool_df)}")
    st.markdown(f"**Failure rate:** {n_failures / len(pool_df) * 100:.1f}%")

    if "failure_modes" in pool_df.columns:
        st.markdown("**Failure mode breakdown:**")
        modes = pool_df["failure_modes"].dropna().replace("", pd.NA).dropna()
        all_modes = []
        for m in modes:
            all_modes.extend(m.split(","))
        if all_modes:
            mode_counts = pd.Series(all_modes).value_counts()
            st.bar_chart(mode_counts)
