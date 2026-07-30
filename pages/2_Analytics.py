"""
Analytics page — Quality analytics dashboard reading from defects_data.csv.
Shows defect trends, severity distribution, repair costs, and failure mode
breakdowns. This is the "quality analytics" panel from the 3-dataset flow.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "packages"))

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Quality Analytics", layout="wide")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
DEFECTS_PATH = os.path.join(DATA_DIR, "defects_data.csv")
AI4I_PATH = os.path.join(DATA_DIR, "ai4i2020.csv")
DATASET_PATH = os.path.join(DATA_DIR, "dataset.csv")

st.title("📊 Quality Analytics")
st.caption(
    "Historical analysis of defect records, failure modes, and repair costs "
    "from the manufacturing production data."
)


@st.cache_data
def load_defects():
    if os.path.exists(DEFECTS_PATH):
        return pd.read_csv(DEFECTS_PATH)
    return pd.DataFrame()


@st.cache_data
def load_ai4i():
    if os.path.exists(AI4I_PATH):
        return pd.read_csv(AI4I_PATH)
    return pd.DataFrame()


@st.cache_data
def load_dataset():
    if os.path.exists(DATASET_PATH):
        return pd.read_csv(DATASET_PATH)
    return pd.DataFrame()


defects_df = load_defects()
ai4i_df = load_ai4i()
dataset_df = load_dataset()

if defects_df.empty:
    st.warning("No defect data found. Run `python3 generate_dataset.py` first.")
    st.stop()


# === Section 1: Overview KPIs ===
st.markdown("---")
kpi1, kpi2, kpi3, kpi4 = st.columns(4)

total_defects = len(defects_df)
total_cost = defects_df["repair_cost"].sum()
avg_cost = defects_df["repair_cost"].mean()
critical_count = (defects_df["severity"] == "Critical").sum()

kpi1.metric("🔧 Total Defects", f"{total_defects:,}")
kpi2.metric("💰 Total Repair Cost", f"${total_cost:,.0f}")
kpi3.metric("📊 Avg Repair Cost", f"${avg_cost:,.0f}")
kpi4.metric("🚨 Critical Defects", f"{critical_count:,}")


# === Section 2: Defect Type Distribution ===
st.markdown("---")
col1, col2 = st.columns(2)

with col1:
    st.subheader("🏷️ Defect Type Distribution")
    defect_counts = defects_df["defect_type"].value_counts()
    st.bar_chart(defect_counts)

with col2:
    st.subheader("⚠️ Severity Breakdown")
    severity_counts = defects_df["severity"].value_counts()
    # Order: Critical, Major, Minor
    severity_order = ["Critical", "Major", "Minor"]
    severity_counts = severity_counts.reindex(severity_order).fillna(0)
    st.bar_chart(severity_counts)


# === Section 3: Failure Mode Analysis ===
st.markdown("---")
col3, col4 = st.columns(2)

with col3:
    st.subheader("🔬 Failure Mode Distribution")
    if "failure_mode" in defects_df.columns:
        mode_counts = defects_df["failure_mode"].value_counts()
        st.bar_chart(mode_counts)

        st.markdown("**Failure mode key:**")
        mode_legend = {
            "TWF": "Tool Wear Failure — tool replacement triggered",
            "HDF": "Heat Dissipation Failure — insufficient cooling",
            "PWF": "Power Failure — power outside 3500-9000W range",
            "OSF": "Overstrain Failure — excessive tool stress",
            "RNF": "Random Failure — no specific cause identified",
        }
        for code, desc in mode_legend.items():
            st.markdown(f"- **{code}**: {desc}")

with col4:
    st.subheader("💰 Repair Cost by Defect Type")
    cost_by_type = defects_df.groupby("defect_type")["repair_cost"].agg(["mean", "sum", "count"])
    cost_by_type.columns = ["Avg Cost ($)", "Total Cost ($)", "Count"]
    cost_by_type = cost_by_type.sort_values("Total Cost ($)", ascending=False)
    st.dataframe(cost_by_type.style.format({
        "Avg Cost ($)": "${:.0f}",
        "Total Cost ($)": "${:,.0f}",
    }), use_container_width=True)


# === Section 4: Machine-level analysis ===
st.markdown("---")
st.subheader("🏭 Defects by Machine")

if "machine_id" in defects_df.columns:
    machine_defects = defects_df.groupby("machine_id").agg(
        defect_count=("defect_id", "count"),
        total_cost=("repair_cost", "sum"),
        critical_count=("severity", lambda x: (x == "Critical").sum()),
    ).sort_values("defect_count", ascending=False)

    machine_defects.columns = ["Defect Count", "Total Cost ($)", "Critical Count"]
    st.dataframe(machine_defects.style.format({
        "Total Cost ($)": "${:,.0f}",
    }), use_container_width=True)


# === Section 5: Inspection Method Effectiveness ===
st.markdown("---")
col5, col6 = st.columns(2)

with col5:
    st.subheader("🔍 Inspection Method Usage")
    inspection_counts = defects_df["inspection_method"].value_counts()
    st.bar_chart(inspection_counts)

with col6:
    st.subheader("📍 Defect Location Hotspots")
    location_counts = defects_df["defect_location"].value_counts().head(10)
    st.bar_chart(location_counts)


# === Section 6: AI4I2020 Dataset Overview ===
if not ai4i_df.empty:
    st.markdown("---")
    st.subheader("📈 AI4I2020 Training Dataset Overview")

    overview1, overview2, overview3 = st.columns(3)
    with overview1:
        st.metric("Total Records", f"{len(ai4i_df):,}")
    with overview2:
        failure_rate = ai4i_df["Machine failure"].mean() * 100
        st.metric("Failure Rate", f"{failure_rate:.1f}%")
    with overview3:
        type_dist = ai4i_df["Type"].value_counts().to_dict()
        st.metric("Product Types", f"L:{type_dist.get('L', 0)} M:{type_dist.get('M', 0)} H:{type_dist.get('H', 0)}")

    # Sensor distributions
    st.markdown("**Sensor Reading Distributions (Training Data)**")
    sensor_cols = [
        "Air temperature [K]", "Process temperature [K]",
        "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]",
    ]
    tabs = st.tabs(sensor_cols)
    for tab, col in zip(tabs, sensor_cols):
        with tab:
            st.line_chart(ai4i_df[col].reset_index(drop=True))


# === Section 7: Machine Operation Log Overview ===
if not dataset_df.empty:
    st.markdown("---")
    st.subheader("⚙️ Machine Operation Log (dataset.csv)")

    op1, op2, op3 = st.columns(3)
    with op1:
        status_counts = dataset_df["Machine_Status"].value_counts()
        st.markdown("**Machine Status**")
        st.bar_chart(status_counts)
    with op2:
        qc_counts = dataset_df["Quality_Check"].value_counts()
        st.metric("Quality Pass Rate", f"{qc_counts.get(True, 0) / len(dataset_df) * 100:.1f}%")
    with op3:
        st.metric("Total Products", f"{dataset_df['Product_Count'].max():,}")


# === Section 8: Raw data viewer ===
st.markdown("---")
with st.expander("📄 View Raw Defect Records"):
    st.dataframe(defects_df, use_container_width=True)

with st.expander("📄 View Raw AI4I2020 Data (first 100 rows)"):
    if not ai4i_df.empty:
        st.dataframe(ai4i_df.head(100), use_container_width=True)

with st.expander("📄 View Raw Machine Operation Log (first 100 rows)"):
    if not dataset_df.empty:
        st.dataframe(dataset_df.head(100), use_container_width=True)
