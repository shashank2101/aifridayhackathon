# Project walkthrough — AI-based manufacturing defect prediction

This is the plain-English explanation of everything happening in this project: what
the data actually looks like, what we've already built, and exactly what's left to
build and how — so anyone on the team can read this once and understand the whole
system without digging through code first.

---

## 1. What problem are we actually solving

Production logs contain process readings for every batch (temperature, pressure,
speed, etc). Some batches turn out defective. We want a system that looks at a
batch's readings *before* it's inspected and says: "this one looks risky, here's
probably why, here's what to check."

Two separate jobs live inside that sentence:
1. **Is this batch risky?** — a numbers problem → handled by a trained ML model.
2. **Why, and what should the operator do?** — a reasoning/explanation problem →
   handled by an LLM (Gemini or Ollama), but only called when step 1 flags something.

---

## 2. The 3-Dataset Flow (AI4I2020 Standard)

To make this demo highly credible, we use the industry-standard **AI4I 2020 Predictive Maintenance Dataset** physics, split into a realistic 3-dataset pipeline:

1. **`ai4i2020.csv`**: The core predictive maintenance dataset. This contains the exact sensor readings (Temperature, RPM, Torque, Tool Wear) and the 5 specific failure modes (TWF, HDF, PWF, OSF, RNF). **The ML model trains exclusively on this file.**
2. **`dataset.csv`**: The machine operation log. This is a real-time time-series view that adds timestamps, machine status, and product counts.
3. **`defects_data.csv`**: The quality inspection records. When a failure happens, this dataset logs a corresponding defect, adding realistic repair costs ($50–$2500), defect severity, and inspection methods based on the specific failure mode.

*There is also a 4th dataset called `log_pool.csv` which is purely for the live demo. It's a subset of data that the model has **never seen**, which the "Producer" page pushes to simulate live incoming batches.*

### The Data
Every row is one production batch. The model learns that specific combinations (like high torque combined with high tool wear) cause specific failures (Overstrain Failure).

---

## 3. What we've built

### Step 1 — Generate a realistic synthetic dataset (`generate_dataset.py`)
We simulate realistic AI4I2020 physics. For example, Power Failures (PWF) trigger exactly when `Torque × RPM × (2π/60)` falls outside the safe 3500-9000W range. Output:
- `ai4i2020.csv`, `dataset.csv`, `defects_data.csv`
- `log_pool.csv` (for the live demo)
- `spec_limits.json` — the allowed range for every parameter.

### Step 2 — Train the ML risk model (`train_model.py`)
We train a **Random Forest classifier** on `ai4i2020.csv`: it learns which combinations of parameter values tend to precede a failed/reworked batch. It uses engineered physics features like `temp_diff` and `power_W` to make this easier.

Result, validated on a 20% holdout set:
- **Accuracy**: 94.8%
- **Recall (catching failures)**: 98%

Saved as `model/risk_model.joblib` — this gets loaded once when the app starts.

### Step 3 — The LangGraph pipeline (`graph.py`)
This runs every time a new batch arrives from the live queue. It's a small graph with 5 nodes:
- `preprocess`: Calculates engineered features and deviations.
- `score`: Asks the Random Forest for a risk score.
- `route`: Branches to the LLM if risk is high, otherwise skips.
- `reason`: Calls the LLM (Gemini or Ollama) with context to get a plain-English explanation.
- `format`: Returns the final alert dict.

### Step 4 — Three Streamlit pages (`app.py`)
- **Dashboard (`app.py`)**: Auto-refreshes every few seconds, reads any new rows from the queue, runs them through LangGraph, and renders the risk table and AI drill-down.
- **Producer (`pages/1_Producer.py`)**: Loads `log_pool.csv`, has a "Push next batch" button that appends rows to a shared queue simulating live incoming MES data.
- **Analytics (`pages/2_Analytics.py`)**: Reads the `defects_data.csv` to provide high-level quality analytics, charting repair costs, defect severities, and failure modes across the factory.

---

## 4. File structure

```
project/
  app.py                     # Streamlit dashboard (main page)
  pages/
    1_Producer.py             # simulate live MES push
    2_Analytics.py             # quality analytics dashboard
  graph.py                    # LangGraph pipeline
  train_model.py               # ML model trainer
  generate_dataset.py           # dataset generator
  data/
    ai4i2020.csv                # predictive maintenance data (model trains here)
    dataset.csv                  # machine operation log
    defects_data.csv              # quality/repair records
    log_pool.csv                   # held-out batches for live demo
    spec_limits.json                # parameter constraints
  model/
    risk_model.joblib             # trained classifier
    feature_importance.json        # top features
  requirements.txt
```
