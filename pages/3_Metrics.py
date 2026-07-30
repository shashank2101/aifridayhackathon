"""
Metrics & ROI Dashboard — The explicit "Success Metrics" required by the judges.
Shows model performance, projected defect reduction (ROI), and user engagement.
"""

import os
import json
import pandas as pd
import numpy as np
import streamlit as st

st.set_page_config(page_title="Success Metrics & ROI", layout="wide")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "model")

METRICS_PATH = os.path.join(MODEL_DIR, "metrics.json")
DEFECTS_PATH = os.path.join(DATA_DIR, "defects_data.csv")
FEEDBACK_PATH = os.path.join(DATA_DIR, "feedback.csv")
POOL_PATH = os.path.join(DATA_DIR, "log_pool.csv")

st.title("📈 Success Metrics & ROI")
st.caption("Tracking the three core hackathon success criteria: Prediction Accuracy, Defect Reduction, and User Engagement.")

# --- 1. Model Accuracy ---
st.markdown("## 1. Prediction Accuracy")

if os.path.exists(METRICS_PATH):
    with open(METRICS_PATH, "r") as f:
        metrics = json.load(f)
    
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Model Accuracy", f"{metrics.get('accuracy', 0) * 100:.1f}%")
    m2.metric("Precision", f"{metrics.get('precision', 0) * 100:.1f}%")
    m3.metric("Recall (Failures Caught)", f"{metrics.get('recall', 0) * 100:.1f}%")
    m4.metric("F1 Score", f"{metrics.get('f1', 0) * 100:.1f}%")

    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown("**Confusion Matrix (Test Set)**")
        cm = metrics.get("confusion_matrix", [[0,0],[0,0]])
        cm_df = pd.DataFrame(cm, index=["Actual Normal", "Actual Failure"], columns=["Pred Normal", "Pred Failure"])
        st.dataframe(cm_df, use_container_width=True)
        st.caption("A high recall tradeoff is intentional for maintenance: false alarms are cheaper than missed failures.")
        
    with col2:
        st.markdown("**Top Predictive Features**")
        importances = metrics.get("feature_importance", {})
        if importances:
            imp_df = pd.DataFrame(list(importances.items()), columns=["Feature", "Importance"])
            st.bar_chart(imp_df.set_index("Feature"))
else:
    st.info("Metrics not found. Run `python3 train_model.py` to generate them.")


# --- 2. Defect Reduction / ROI ---
st.markdown("---")
st.markdown("## 2. Defect Reduction & Cost Impact")

@st.cache_data
def get_roi_stats():
    # Calculate average cost from historical defects
    avg_cost = 912  # default fallback
    if os.path.exists(DEFECTS_PATH):
        df = pd.read_csv(DEFECTS_PATH)
        if not df.empty and "repair_cost" in df.columns:
            avg_cost = df["repair_cost"].mean()
            
    # Simulate ROI using the unseen log_pool
    if os.path.exists(POOL_PATH) and os.path.exists(METRICS_PATH):
        pool = pd.read_csv(POOL_PATH)
        failures = (pool["ground_truth_label"] == "FAILURE").sum()
        
        # Apply the model's recall to estimate caught failures
        with open(METRICS_PATH, "r") as f:
            m = json.load(f)
        recall = m.get("recall", 0.95)
        
        caught = int(failures * recall)
        missed = failures - caught
        
        saved_cost = caught * avg_cost
        exposed_cost = missed * avg_cost
        
        return {
            "total_failures": failures,
            "caught": caught,
            "missed": missed,
            "saved_cost": saved_cost,
            "exposed_cost": exposed_cost,
            "avg_cost": avg_cost
        }
    return None

roi = get_roi_stats()

if roi:
    c1, c2, c3 = st.columns(3)
    c1.metric("Failures Prevented (Simulated)", f"{roi['caught']} / {roi['total_failures']}")
    c2.metric("Estimated Cost Saved", f"${roi['saved_cost']:,.0f}", f"+${roi['saved_cost']:,.0f}")
    c3.metric("Residual Risk Exposure", f"${roi['exposed_cost']:,.0f}", f"-${roi['saved_cost']:,.0f}", delta_color="inverse")
    
    st.info(f"💡 **Business Case**: Based on a historical average repair cost of **${roi['avg_cost']:,.0f}**, applying this AI agent to the current production queue would save **${roi['saved_cost']:,.0f}** by catching {roi['caught']} defects before they occur.")
else:
    st.info("Run the simulation first to see defect reduction metrics.")


# --- 3. User Engagement ---
st.markdown("---")
st.markdown("## 3. User Engagement & Feedback Loop")

if os.path.exists(FEEDBACK_PATH):
    fb_df = pd.read_csv(FEEDBACK_PATH)
    if not fb_df.empty:
        e1, e2, e3 = st.columns(3)
        total_fb = len(fb_df)
        upvotes = (fb_df["feedback"] == "up").sum()
        corrections = fb_df["correction"].notna().sum() - (fb_df["correction"] == "").sum()
        
        e1.metric("Total Operator Ratings", total_fb)
        e2.metric("Helpful Ratio", f"{(upvotes/total_fb)*100:.0f}%")
        e3.metric("Written Corrections Submitted", corrections)
        
        st.markdown("**Recent Operator Feedback**")
        st.dataframe(fb_df, use_container_width=True)
    else:
        st.info("No feedback collected yet. Rate alerts on the Dashboard to see engagement metrics.")
else:
    st.info("No feedback collected yet. Rate alerts on the Dashboard to see engagement metrics.")
