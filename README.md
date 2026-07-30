# Manufacturing Defect Risk Platform

AI Friday Season 2 Qualifiers — AI-Based Manufacturing Defect Prediction from
Production Logs, aligned to the **AI4I 2020 Predictive Maintenance Dataset** standard.

See `docs/project_walkthrough.md` for the full explanation and
`docs/architecture.pdf` for the system diagram.

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

This opens the **Dashboard** page. Use the **Producer** page (left sidebar) to
simulate new batches arriving from the MES feed — click "Push next batch", then
watch the Dashboard catch anomalies in real time.

The **Analytics** page shows historical quality analytics across all 3 datasets.

## Local AI Reasoning (Ollama)

This project is built for **100% local execution** using Ollama to ensure data privacy and operational security. Low-risk batches are handled purely by the ML model; high-risk batches are routed to the local LLM.

Make sure Ollama is installed and running locally:

```bash
ollama pull llama3.2:3b
ollama serve
```

*(Note: The `llm_client.py` file also contains a backup configuration for the TCS GenAI Lab endpoint, but Ollama is the active default).*

## Datasets (3-dataset flow)

This project uses 3 interconnected datasets aligned to an industrial
manufacturing pipeline:

| File | Purpose | Key Columns |
|---|---|---|
| `ai4i2020.csv` | Predictive maintenance (ML training) | UDI, Product ID, Type, Air/Process Temp, RPM, Torque, Tool Wear, Machine failure, TWF/HDF/PWF/OSF/RNF |
| `dataset.csv` | Machine operation log | ID, Timestamp, Machine_ID, Sensors, Machine_Status, Quality_Check, Product_Count |
| `defects_data.csv` | Defect/repair records | defect_id, product_id, defect_type, severity, repair_cost, failure_mode |

### How they connect

```
Machine Telemetry (dataset.csv)
       │
       ▼
Machine Health Assessment
       │
       ▼
Predictive Maintenance (ai4i2020.csv)
       │
       ▼
Failure Prediction (ML Model)
       │
       ▼
Quality Inspection
       │
       ▼
Defect Records (defects_data.csv)
       │
       ▼
Analytics & Cost Optimization
```

### Failure modes (real AI4I2020 rules)

| Mode | Rule |
|---|---|
| **TWF** | Tool wear randomly triggers replacement between 200-240 min |
| **HDF** | (process_temp - air_temp) < 8.6K AND rpm < 1380 |
| **PWF** | power < 3500W or > 9000W |
| **OSF** | tool_wear × torque > threshold (11K/12K/13K by type L/M/H) |
| **RNF** | 0.1% random chance |

## What's already trained

`model/risk_model.joblib` is a Random Forest trained on `data/ai4i2020.csv`
with physics-based engineered features. To regenerate:

```bash
python3 generate_dataset.py   # generates all 3 CSVs + log_pool.csv
python3 train_model.py         # retrains and validates the model
```

## Project structure

```
app.py                          # Streamlit dashboard (main entrypoint)
pages/
  1_Producer.py                  # simulates the live MES feed
  2_Analytics.py                  # quality analytics from defects_data.csv
  3_Metrics.py                    # Success metrics, prediction accuracy & ROI
graph.py                         # LangGraph pipeline: preprocess -> ML score -> guardrails -> LLM -> alert
train_model.py                    # trains the Random Forest on ai4i2020.csv
generate_dataset.py                # generates all synthetic datasets
requirements.txt
data/
  ai4i2020.csv                    # predictive maintenance dataset (model trains on this)
  dataset.csv                      # machine operation log
  defects_data.csv                  # defect/repair records
  log_pool.csv                      # held-out batches for live demo (unseen by model)
  spec_limits.json                   # normal operating ranges
  live_queue.csv                     # created at runtime by Producer
  alerts.csv                          # created at runtime by Dashboard
  feedback.csv                         # created at runtime from feedback buttons
model/
  risk_model.joblib                # trained Random Forest classifier
  feature_importance.json           # feature importances
docs/
  architecture.pdf / .md           # system architecture with guardrails
  architecture_diagram.png
  project_walkthrough.md
```

## Testing the pipeline without the UI

```bash
python3 graph.py
```

Runs a hardcoded high-risk test batch through the full pipeline and prints the
resulting alert JSON.
