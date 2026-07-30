"""
Dashboard - the main Streamlit entrypoint. Polls the shared live queue for
batches the Producer page has pushed, runs each new one through the
LangGraph pipeline (graph.py), and renders the resulting risk alerts.

Aligned to AI4I 2020 Predictive Maintenance Dataset standard.

Run with:  streamlit run app.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "packages"))

import time
import pandas as pd
import streamlit as st

from graph import process_batch
st.set_page_config(
    page_title="AI Manufacturing Defect Prediction",
    page_icon="🏭",
    layout="wide",
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
QUEUE_PATH = os.path.join(DATA_DIR, "live_queue.csv")
ALERTS_PATH = os.path.join(DATA_DIR, "alerts.csv")
FEEDBACK_PATH = os.path.join(DATA_DIR, "feedback.csv")
PROCESSED_MARKER = os.path.join(DATA_DIR, ".processed_batch_ids.txt")

ALERT_COLUMNS = [
    "batch_id", "timestamp", "machine_id", "product_type",
    "risk_score", "risk_level",
    "air_temp_K", "process_temp_K", "rpm", "torque_Nm",
    "tool_wear_min", "power_W", "temp_diff",
    "deviated_params", "probable_cause", "recommended_action",
    "confidence", "needs_human_review",
]

# --- Sidebar ---
with st.sidebar:
    st.image("https://img.icons8.com/color/96/factory.png", width=60)
    st.subheader("⚙️ Configuration")

    st.sidebar.markdown("### ⚙️ LLM Configuration")
    st.sidebar.info("The AI reasoning backend is now hardcoded in `graph.py`. Open `graph.py` to comment/uncomment the client you want to use (Ollama is set as the default).")
    
    st.divider()
    st.sidebar.markdown("### 🎛️ Pipeline Tuning")
    risk_thresh = st.slider("LLM Routing Threshold", min_value=0.1, max_value=0.9, value=0.5, step=0.05,
                            help="Only batches with risk >= this threshold will trigger the LLM reasoning agent.")
    os.environ["RISK_THRESHOLD"] = str(risk_thresh)
    
    st.divider()
    auto_refresh = st.toggle("Enable auto-refresh", value=True, help="Disable this while chatting to prevent interruptions.")
    refresh_seconds = st.slider("Auto-refresh interval (seconds)", 3, 30, 5, disabled=not auto_refresh)
    st.info("Open the **Producer** page (left sidebar) to push new batches.")


# --- Header ---
st.title("🏭 Live Defect Risk Dashboard")
st.caption(
    "Reads new batches pushed from the Producer page, scores them with the "
    "ML model (Random Forest on AI4I2020 features), and calls the LLM for "
    "an explanation when risk is high."
)


# --- Helpers ---

def load_processed_ids() -> set:
    if os.path.exists(PROCESSED_MARKER):
        with open(PROCESSED_MARKER) as f:
            return {line.strip() for line in f if line.strip()}
    return set()


def mark_processed(batch_id: str):
    with open(PROCESSED_MARKER, "a") as f:
        f.write(f"{batch_id}\n")


def load_alerts() -> pd.DataFrame:
    if os.path.exists(ALERTS_PATH):
        return pd.read_csv(ALERTS_PATH)
    return pd.DataFrame(columns=ALERT_COLUMNS)


def append_alert(alert: dict):
    df = pd.DataFrame([alert])
    header = not os.path.exists(ALERTS_PATH)
    df.to_csv(ALERTS_PATH, mode="a", header=header, index=False)


def process_new_batches() -> int:
    if not os.path.exists(QUEUE_PATH):
        return 0
    queue_df = pd.read_csv(QUEUE_PATH)

    # Use Product ID as the batch identifier
    id_col = "Product ID" if "Product ID" in queue_df.columns else "batch_id"
    processed_ids = load_processed_ids()
    new_rows = queue_df[~queue_df[id_col].isin(processed_ids)]

    for _, row in new_rows.iterrows():
        alert = process_batch(row.to_dict())
        append_alert(alert)
        mark_processed(row[id_col])
    return len(new_rows)


# --- Process new batches ---
with st.spinner("🤖 AI is analyzing new telemetry for root causes..."):
    n_new = process_new_batches()
if n_new:
    st.toast(f"✅ Processed {n_new} new batch(es)")

alerts_df = load_alerts()

# --- KPIs ---
st.markdown("---")
kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("📦 Total Processed", len(alerts_df))
kpi2.metric(
    "🔴 High Risk",
    int((alerts_df["risk_level"] == "High").sum()) if len(alerts_df) else 0,
)
kpi3.metric(
    "🟡 Medium Risk",
    int((alerts_df["risk_level"] == "Medium").sum()) if len(alerts_df) else 0,
)
kpi4.metric(
    "👁️ Needs Review",
    int(alerts_df["needs_human_review"].sum()) if len(alerts_df) else 0,
)

# --- Risk table ---
st.markdown("---")
st.subheader("📋 Batch Risk Table")

RISK_COLORS = {"High": "#F7C1C1", "Medium": "#FAC775", "Low": "#C0DD97"}


def highlight_risk(row):
    color = RISK_COLORS.get(row["risk_level"], "")
    return [f"background-color: {color}"] * len(row)


if len(alerts_df):
    display_df = alerts_df.sort_values("timestamp", ascending=False)

    # Show compact view with key columns
    compact_cols = [
        "batch_id", "timestamp", "machine_id", "product_type",
        "risk_score", "risk_level", "deviated_params",
    ]
    available_cols = [c for c in compact_cols if c in display_df.columns]

    st.dataframe(
        display_df[available_cols].style.apply(highlight_risk, axis=1),
        use_container_width=True, height=320,
    )
else:
    st.info("No batches processed yet. Push a batch from the **Producer** page.")

# --- Drill-down ---
st.markdown("---")
st.subheader("🔍 Batch Drill-Down")

if len(alerts_df):
    risky_df = alerts_df[alerts_df["risk_level"].isin(["High", "Medium"])]
    if len(risky_df) == 0:
        st.success("No risky batches detected yet! All processed batches are Low risk.")
    else:
        selected = st.selectbox("Select a risky batch to investigate", risky_df["batch_id"].tolist())
        row = risky_df[risky_df["batch_id"] == selected].iloc[0]

    if len(risky_df) > 0:
        col_left, col_right = st.columns([3, 2])
        with col_left:
            st.markdown(f"### Risk Assessment")
            risk_emoji = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}.get(row["risk_level"], "⚪")
            st.markdown(f"**Risk level:** {risk_emoji} {row['risk_level']}  (score: {row['risk_score']:.3f})")
            st.markdown(f"**Product type:** {row.get('product_type', 'N/A')}")
            st.markdown(f"**Machine:** {row.get('machine_id', 'N/A')}")
            st.markdown(f"**Deviated parameters:** {row.get('deviated_params') or 'none'}")

            if row.get("probable_cause"):
                if str(row['probable_cause']).startswith("Reasoning call failed"):
                    st.error(f"**🤖 AI Error:** {row['probable_cause']}")
                    
                    if st.button("🔄 Retry AI Analysis"):
                        with st.spinner("Re-running AI Analysis..."):
                            from graph import process_batch
                            # Get original data
                            queue_df = pd.read_csv(QUEUE_PATH)
                            id_col = "Product ID" if "Product ID" in queue_df.columns else "batch_id"
                            batch_data = queue_df[queue_df[id_col] == selected].iloc[0].to_dict()
                            
                            # Rerun pipeline
                            new_alert = process_batch(batch_data)
                            
                            # Replace old alert
                            alerts = pd.read_csv(ALERTS_PATH)
                            alerts = alerts[alerts["batch_id"] != selected]
                            alerts = pd.concat([alerts, pd.DataFrame([new_alert])], ignore_index=True)
                            alerts.to_csv(ALERTS_PATH, index=False)
                            st.rerun()
                else:
                    st.markdown(f"**🤖 Probable cause:** {row['probable_cause']}")
                    st.markdown(f"**🔧 Recommended action:** {row['recommended_action']}")
                    st.markdown(f"**Confidence:** {row['confidence']}")

            if row.get("needs_human_review"):
                st.warning("⚠️ Flagged for human review by the output guardrail.")

        with col_right:
            st.markdown("### Sensor Readings")
            sensor_data = {
                "Air Temp (K)": row.get("air_temp_K", "N/A"),
                "Process Temp (K)": row.get("process_temp_K", "N/A"),
                "RPM": row.get("rpm", "N/A"),
                "Torque (Nm)": row.get("torque_Nm", "N/A"),
                "Tool Wear (min)": row.get("tool_wear_min", "N/A"),
                "Power (W)": row.get("power_W", "N/A"),
                "Temp Diff (K)": row.get("temp_diff", "N/A"),
            }
            for label, val in sensor_data.items():
                st.markdown(f"**{label}:** {val}")

            st.markdown("---")
            st.markdown("**Was this alert helpful?**")
            fb1, fb2 = st.columns(2)

            def log_feedback(feedback_value, correction=""):
                entry = {"batch_id": selected, "feedback": feedback_value, "correction": correction}
                pd.DataFrame([entry]).to_csv(
                    FEEDBACK_PATH, mode="a", header=not os.path.exists(FEEDBACK_PATH), index=False
                )
                st.success("Thanks for the feedback!")

            if fb1.button("👍 Helpful", key=f"up_{selected}"):
                log_feedback("up")
            
            with fb2.popover("👎 Not helpful"):
                corr_text = st.text_input("What was the actual issue?", key=f"corr_{selected}")
                if st.button("Submit Correction", key=f"sub_{selected}"):
                    log_feedback("down", corr_text)

        # --- Chatbot for this specific batch ---
        st.markdown("---")
        st.subheader(f"💬 Chat with AI Assistant about {selected}")
        st.caption("Ask follow-up questions about this specific batch. The AI has access to its sensor readings and risk profile.")

        chat_key = f"chat_{selected}"
        if chat_key not in st.session_state:
            st.session_state[chat_key] = []

        for msg in st.session_state[chat_key]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        if prompt := st.chat_input(f"Ask a question about batch {selected}..."):
            st.session_state[chat_key].append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                context = f"""
                You are a manufacturing AI assistant helping an operator. The user is asking about batch {selected}.
                Risk level: {row['risk_level']} (Score: {row['risk_score']})
                Deviated parameters: {row.get('deviated_params', 'none')}
                Probable cause determined earlier: {row.get('probable_cause', 'unknown')}
                Sensor readings: Air Temp {row.get('air_temp_K')}K, Process Temp {row.get('process_temp_K')}K, RPM {row.get('rpm')}, Torque {row.get('torque_Nm')}Nm, Wear {row.get('tool_wear_min')}min.
                Engineered features: Power {row.get('power_W')}W, Temp Diff {row.get('temp_diff')}K.
            
                User question: {prompt}
            
                Please provide a helpful, concise answer based on this context. Do not use markdown code blocks unless necessary.
                """
            
                with st.spinner("Thinking..."):
                    try:
                        from llm_client import llm_call
                        response_text = llm_call(context)
                        
                        st.markdown(response_text)
                        st.session_state[chat_key].append({"role": "assistant", "content": response_text})
                    except Exception as e:
                        st.error(f"Error calling AI: {e}")

else:
    st.caption("Drill-down will appear here once a batch has been processed.")

# --- Auto-refresh ---
if auto_refresh:
    time.sleep(refresh_seconds)
    st.rerun()
