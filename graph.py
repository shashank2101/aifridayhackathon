"""
LangGraph pipeline that turns one incoming production batch into a finished
risk alert. This is the "AI reasoning agent" component from architecture.pdf.

Flow:
    preprocess -> score -> [router] -> reason (guardrailed Gemini call) -> format
                                 (or)-> skip (low risk, no LLM call needed) -/

Import process_batch() from this module and call it once per new row.

Supports both Gemini API and local Ollama (llama3.2:3b) — toggle via
REASONING_BACKEND env var ("gemini" or "ollama").
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from langgraph.graph import StateGraph, END

# --- Optional: uncomment to use Ollama instead of Gemini ---
# import requests  # for Ollama HTTP API

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Raw sensor features from AI4I2020
RAW_FEATURES = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

# Engineered features (computed at inference time, same as training)
ENGINEERED_FEATURES = [
    "temp_diff", "power_W", "overstrain_product", "torque_speed_ratio",
]

ALL_FEATURES = RAW_FEATURES + ENGINEERED_FEATURES

# Anything scoring at or above this is routed to the LLM reasoning node.
def get_risk_threshold():
    return float(os.environ.get("RISK_THRESHOLD", 0.5))

with open(os.path.join(BASE_DIR, "data", "spec_limits.json")) as f:
    SPEC_LIMITS = json.load(f)

_risk_model = joblib.load(os.path.join(BASE_DIR, "model", "risk_model.joblib"))

from llm_client import llm_call


def engineer_features_for_row(row: dict) -> dict:
    """Compute the same engineered features used during training."""
    air_temp = row.get("Air temperature [K]", 300)
    proc_temp = row.get("Process temperature [K]", 310)
    rpm = row.get("Rotational speed [rpm]", 2860)
    torque = row.get("Torque [Nm]", 40)
    tool_wear = row.get("Tool wear [min]", 0)

    return {
        "temp_diff": proc_temp - air_temp,
        "power_W": torque * rpm * 2 * np.pi / 60,
        "overstrain_product": tool_wear * torque,
        "torque_speed_ratio": torque / (rpm + 1),
    }


# ---------------------------------------------------------------------------
# Node 1: preprocessing
# ---------------------------------------------------------------------------

def compute_deviations(row: dict) -> dict:
    """Returns only the parameters that breached spec, with how far off
    they were. Empty dict means nothing was out of spec."""
    deviations = {}
    for param, limits in SPEC_LIMITS.items():
        val = row.get(param)
        if val is None:
            continue
        low, high = limits["low"], limits["high"]
        span = high - low
        if val < low:
            deviations[param] = {
                "value": val, "limit": low, "direction": "below",
                "pct_off_range": round((low - val) / span * 100, 1),
            }
        elif val > high:
            deviations[param] = {
                "value": val, "limit": high, "direction": "above",
                "pct_off_range": round((val - high) / span * 100, 1),
            }
    return deviations


def preprocess_node(state: dict) -> dict:
    row = state["batch"]
    # Compute engineered features and add them to the batch
    eng = engineer_features_for_row(row)
    row.update(eng)
    state["batch"] = row
    state["deviations"] = compute_deviations(row)
    return state


# ---------------------------------------------------------------------------
# Node 2: ML risk scoring (no LLM involved)
# ---------------------------------------------------------------------------

def risk_scoring_node(state: dict) -> dict:
    row = state["batch"]
    X = pd.DataFrame([{col: row[col] for col in ALL_FEATURES}])

    proba = _risk_model.predict_proba(X)[0]
    classes = list(_risk_model.classes_)
    failure_idx = classes.index(1) if 1 in classes else int(proba.argmax())
    risk_score = float(proba[failure_idx])

    top_features = dict(
        sorted(
            zip(ALL_FEATURES, _risk_model.feature_importances_.round(3)),
            key=lambda x: -x[1],
        )[:5]
    )

    state["risk_score"] = risk_score
    state["top_features"] = top_features
    return state


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

def route_after_scoring(state: dict) -> str:
    return "reason" if state["risk_score"] >= get_risk_threshold() else "skip"


# ---------------------------------------------------------------------------
# Guardrails
# ---------------------------------------------------------------------------

def input_relevancy_guardrail(row: dict) -> bool:
    """Confirms the payload is legitimate in-scope manufacturing telemetry
    before anything is sent to the LLM. Blocks malformed or unexpected
    payloads from ever reaching the reasoning node."""
    if not isinstance(row, dict):
        return False
    if not all(col in row for col in RAW_FEATURES):
        return False
    return all(isinstance(row[col], (int, float)) for col in RAW_FEATURES)


def output_validator_guardrail(parsed: dict) -> bool:
    """Confirms the LLM's response matches the expected schema before it is
    allowed to reach the dashboard."""
    required = {"probable_cause", "recommended_action", "confidence"}
    if not required.issubset(parsed.keys()):
        return False
    if parsed["confidence"] not in ("High", "Medium", "Low"):
        return False
    if not str(parsed["probable_cause"]).strip():
        return False
    if not str(parsed["recommended_action"]).strip():
        return False
    return True


# ---------------------------------------------------------------------------
# Node 3: LLM reasoning (only reached for flagged batches)
# ---------------------------------------------------------------------------

REASONING_PROMPT = """You are a manufacturing quality assistant. A production batch \
was flagged as high risk by an ML model trained on the AI4I 2020 Predictive Maintenance \
Dataset. Explain the likely failure mode and recommend one concrete preventive action, \
based only on the evidence given below.

Batch ID: {batch_id}
Machine: {machine_id}
Product Type: {product_type}
Risk score (0-1, from the ML model): {risk_score:.2f}

Sensor readings:
  Air temperature: {air_temp} K
  Process temperature: {process_temp} K
  Rotational speed: {rpm} rpm
  Torque: {torque} Nm
  Tool wear: {tool_wear} min

Computed indicators:
  Temperature differential: {temp_diff:.1f} K (HDF threshold: < 8.6 K)
  Power output: {power_W:.0f} W (PWF range: 3500-9000 W)
  Overstrain product: {overstrain:.0f} (OSF threshold: {osf_threshold})

Deviated parameters (value vs spec limit): {deviations}
Top contributing factors from ML feature importances: {top_features}

Known failure modes:
  TWF - Tool Wear Failure (tool wear 200-240 min)
  HDF - Heat Dissipation Failure (temp diff < 8.6K AND rpm < 1380)
  PWF - Power Failure (power < 3500W or > 9000W)
  OSF - Overstrain Failure (tool_wear × torque > threshold)
  RNF - Random Failure (0.1% chance)

Respond with ONLY a JSON object, no markdown fences, no extra text:
{{"probable_cause": "one to two sentence plain-language explanation identifying the most likely failure mode(s)", \
"recommended_action": "one concrete preventive action for the operator", \
"confidence": "High, Medium, or Low"}}
"""


def reasoning_node(state: dict) -> dict:
    row = state["batch"]

    if not input_relevancy_guardrail(row):
        state["reasoning"] = {
            "probable_cause": "Input rejected by relevancy guardrail — payload did not "
                               "match expected manufacturing telemetry schema.",
            "recommended_action": "Review the raw payload manually before retrying.",
            "confidence": "Low",
        }
        state["needs_human_review"] = True
        return state

    product_type = row.get("Type", "M")
    osf_threshold = OSF_THRESHOLDS.get(product_type, 12000)

    prompt = REASONING_PROMPT.format(
        batch_id=row.get("Product ID", row.get("batch_id", "unknown")),
        machine_id=row.get("machine_id", "unknown"),
        product_type=product_type,
        risk_score=state["risk_score"],
        air_temp=row.get("Air temperature [K]", "?"),
        process_temp=row.get("Process temperature [K]", "?"),
        rpm=row.get("Rotational speed [rpm]", "?"),
        torque=row.get("Torque [Nm]", "?"),
        tool_wear=row.get("Tool wear [min]", "?"),
        temp_diff=row.get("temp_diff", 0),
        power_W=row.get("power_W", 0),
        overstrain=row.get("overstrain_product", 0),
        osf_threshold=osf_threshold,
        deviations=json.dumps(state["deviations"]),
        top_features=json.dumps(state["top_features"]),
    )

    try:
        text = llm_call(prompt)

        text = text.strip().replace("```json", "").replace("```", "").strip()
        parsed = json.loads(text)
    except Exception as exc:
        parsed = {
            "probable_cause": f"Reasoning call failed ({exc}). Falling back to rule-based note: "
                               f"deviated parameters were {list(state['deviations'].keys())}.",
            "recommended_action": "Review batch manually.",
            "confidence": "Low",
        }

    if not output_validator_guardrail(parsed):
        state["needs_human_review"] = True
        parsed = {
            "probable_cause": parsed.get("probable_cause", "Output failed validation."),
            "recommended_action": parsed.get("recommended_action", "Review batch manually."),
            "confidence": "Low",
        }
    else:
        state["needs_human_review"] = False

    state["reasoning"] = parsed
    return state


def skip_reasoning_node(state: dict) -> dict:
    state["reasoning"] = None
    state["needs_human_review"] = False
    return state


# OSF thresholds needed in reasoning prompt
OSF_THRESHOLDS = {"L": 11000, "M": 12000, "H": 13000}


# ---------------------------------------------------------------------------
# Node 4: format the final alert
# ---------------------------------------------------------------------------

def format_alert_node(state: dict) -> dict:
    row = state["batch"]
    risk_score = state["risk_score"]
    risk_level = "High" if risk_score >= 0.75 else "Medium" if risk_score >= get_risk_threshold() else "Low"
    reasoning = state["reasoning"] or {}

    state["alert"] = {
        "batch_id": row.get("Product ID", row.get("batch_id")),
        "timestamp": row.get("timestamp"),
        "machine_id": row.get("machine_id"),
        "product_type": row.get("Type", ""),
        "risk_score": round(risk_score, 3),
        "risk_level": risk_level,
        "air_temp_K": row.get("Air temperature [K]"),
        "process_temp_K": row.get("Process temperature [K]"),
        "rpm": row.get("Rotational speed [rpm]"),
        "torque_Nm": row.get("Torque [Nm]"),
        "tool_wear_min": row.get("Tool wear [min]"),
        "power_W": round(row.get("power_W", 0), 1),
        "temp_diff": round(row.get("temp_diff", 0), 1),
        "deviated_params": ",".join(state["deviations"].keys()),
        "probable_cause": reasoning.get("probable_cause", ""),
        "recommended_action": reasoning.get("recommended_action", ""),
        "confidence": reasoning.get("confidence", ""),
        "needs_human_review": state.get("needs_human_review", False),
    }
    return state


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------

def build_graph():
    graph = StateGraph(dict)
    graph.add_node("preprocess", preprocess_node)
    graph.add_node("score", risk_scoring_node)
    graph.add_node("reason", reasoning_node)
    graph.add_node("skip", skip_reasoning_node)
    graph.add_node("format", format_alert_node)

    graph.set_entry_point("preprocess")
    graph.add_edge("preprocess", "score")
    graph.add_conditional_edges("score", route_after_scoring, {"reason": "reason", "skip": "skip"})
    graph.add_edge("reason", "format")
    graph.add_edge("skip", "format")
    graph.add_edge("format", END)

    return graph.compile()


_compiled_graph = None


def get_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph


def process_batch(row: dict) -> dict:
    """Runs one batch (dict of column -> value) through the full pipeline
    and returns the finished alert dict."""
    result = get_graph().invoke({"batch": row})
    return result["alert"]


if __name__ == "__main__":
    # Quick smoke test: simulate a high-risk batch (tool wear failure)
    test_row = {
        "Product ID": "TEST-0001",
        "Type": "L",
        "timestamp": "2026-07-30T10:00:00",
        "machine_id": "LINE1-M01",
        "Air temperature [K]": 300.0,
        "Process temperature [K]": 308.0,  # temp_diff = 8.0 < 8.6 → HDF risk
        "Rotational speed [rpm]": 1350,    # < 1380 → HDF trigger
        "Torque [Nm]": 55.0,
        "Tool wear [min]": 210,            # TWF range
    }
    print(json.dumps(process_batch(test_row), indent=2))
