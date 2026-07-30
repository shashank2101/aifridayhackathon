"""
Generates synthetic manufacturing data aligned to the AI4I 2020
Predictive Maintenance Dataset standard (UCI ML Repository).

Uses the REAL published failure-generating formulas:
  TWF - Tool Wear Failure      (random replacement between 200-240 min)
  HDF - Heat Dissipation Failure (temp_diff < 8.6K AND rpm < 1380)
  PWF - Power Failure           (power < 3500W or > 9000W)
  OSF - Overstrain Failure      (tool_wear * torque > threshold by type)
  RNF - Random Failure          (0.1% chance per process)

Outputs (in ./data/):
  ai4i2020.csv       - Core predictive maintenance dataset (model trains on this)
  dataset.csv        - Machine operation log (timestamps, status, product count)
  defects_data.csv   - Defect/repair records (generated from failure events)
  log_pool.csv       - Held-out live-demo pool (UNSEEN by model), for the
                       Producer page to simulate a live MES feed
  spec_limits.json   - Per-parameter normal operating ranges
"""

import os
import json
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

random.seed(42)
np.random.seed(42)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

LINES = ["LINE1", "LINE2", "LINE3"]
MACHINES_PER_LINE = 2
OPERATORS = [f"OP-{100 + i}" for i in range(12)]
MATERIAL_LOTS = [f"MAT-{8800 + i}" for i in range(20)]

# Product type distribution (matches real AI4I2020)
PRODUCT_TYPES = {"L": 0.50, "M": 0.30, "H": 0.20}

# Overstrain thresholds per product type (real AI4I2020 values)
OSF_THRESHOLDS = {"L": 11000, "M": 12000, "H": 13000}

# Defect type mapping from failure modes
FAILURE_TO_DEFECT = {
    "TWF": {"defect_type": "Tool Damage", "locations": ["Cutting Edge", "Tool Holder", "Spindle Interface", "Tool Tip"]},
    "HDF": {"defect_type": "Thermal Defect", "locations": ["Heat Exchanger", "Cooling System", "Motor Housing", "Bearing Assembly"]},
    "PWF": {"defect_type": "Power Defect", "locations": ["Drive Motor", "Power Supply", "VFD Unit", "Electrical Panel"]},
    "OSF": {"defect_type": "Structural Stress", "locations": ["Shaft", "Gearbox", "Chuck Assembly", "Frame Mount"]},
    "RNF": {"defect_type": "Random/Unknown", "locations": ["Various", "Unspecified", "Control Board", "Sensor Array"]},
}

SEVERITY_WEIGHTS = {
    "TWF": {"Critical": 0.30, "Major": 0.50, "Minor": 0.20},
    "HDF": {"Critical": 0.40, "Major": 0.40, "Minor": 0.20},
    "PWF": {"Critical": 0.50, "Major": 0.35, "Minor": 0.15},
    "OSF": {"Critical": 0.45, "Major": 0.40, "Minor": 0.15},
    "RNF": {"Critical": 0.15, "Major": 0.35, "Minor": 0.50},
}

REPAIR_COST_RANGE = {
    "Critical": (800, 2500),
    "Major": (250, 800),
    "Minor": (50, 250),
}

INSPECTION_METHODS = ["Visual", "Automated", "Manual", "Ultrasonic", "X-Ray"]

# Sizes
N_TRAIN = 8000          # rows for ai4i2020.csv (model trains on this)
N_POOL = 500            # rows for log_pool.csv (live demo, unseen by model)

# Normal operating ranges (for spec_limits.json)
SPEC_LIMITS = {
    "Air temperature [K]":     {"low": 295.0, "high": 305.0, "unit": "K"},
    "Process temperature [K]": {"low": 305.0, "high": 315.0, "unit": "K"},
    "Rotational speed [rpm]":  {"low": 1168,  "high": 2886,  "unit": "rpm"},
    "Torque [Nm]":             {"low": 3.8,   "high": 76.2,  "unit": "Nm"},
    "Tool wear [min]":         {"low": 0,     "high": 240,   "unit": "min"},
}


# ---------------------------------------------------------------------------
# Core generation functions (AI4I2020 physics)
# ---------------------------------------------------------------------------

def sample_product_type():
    """Sample product type: L (50%), M (30%), H (20%)."""
    return np.random.choice(
        list(PRODUCT_TYPES.keys()),
        p=list(PRODUCT_TYPES.values()),
    )


def generate_sensor_row(product_type, tool_wear_state):
    """Generate one row of sensor data using AI4I2020 physics.
    
    Distributions are calibrated so that:
      - Power (torque × rpm × 2π/60) is nominally ~6000-7000W
        (safely inside the 3500-9000W PWF safe range)
      - Failures occur at a realistic ~3-5% rate overall
    
    Returns (sensor_dict, failure_dict, updated_tool_wear).
    """
    # Air temperature: N(300, 2) K — ambient conditions
    air_temp = np.random.normal(300, 2)
    air_temp = round(np.clip(air_temp, 293, 307), 1)

    # Process temperature: air_temp + 10 + N(0, 1) — physically coupled
    process_temp = air_temp + 10 + np.random.normal(0, 1)
    process_temp = round(np.clip(process_temp, 303, 317), 1)

    # Rotational speed: centered ~1500 rpm with right-skewed noise
    # At 1500 rpm with 40 Nm torque: power ≈ 1500 × 40 × 2π/60 ≈ 6283W (safely in range)
    rpm = int(np.clip(
        1500 + np.random.normal(0, 200) + np.random.choice([-1, 1]) * np.random.exponential(100),
        1168, 2886
    ))

    # Torque: ~40 Nm, negatively correlated with RPM (power ≈ constant)
    torque = round(np.clip(
        40 + np.random.normal(0, 10) - 0.008 * (rpm - 1500),
        3.8, 76.2
    ), 1)

    # Tool wear: cumulative, H-type wears +5 min/cycle, M +3, L +2
    wear_increment = {"H": 5, "M": 3, "L": 2}[product_type]
    tool_wear = tool_wear_state + wear_increment
    if tool_wear > 240:
        tool_wear = 0  # tool replaced

    # --- Apply AI4I2020 failure rules ---
    twf = 0
    hdf = 0
    pwf = 0
    osf = 0
    rnf = 0

    # TWF: tool randomly fails between 200-240 min of wear
    twf_threshold = random.randint(200, 240)
    if tool_wear >= twf_threshold:
        twf = 1

    # HDF: heat dissipation failure
    temp_diff = process_temp - air_temp
    if temp_diff < 8.6 and rpm < 1380:
        hdf = 1

    # PWF: power failure
    power = torque * rpm * 2 * np.pi / 60  # watts
    if power < 3500 or power > 9000:
        pwf = 1

    # OSF: overstrain failure
    overstrain_product = tool_wear * torque
    if overstrain_product > OSF_THRESHOLDS[product_type]:
        osf = 1

    # RNF: random failure (0.1% chance)
    if random.random() < 0.001:
        rnf = 1

    machine_failure = 1 if (twf or hdf or pwf or osf or rnf) else 0

    sensors = {
        "Air temperature [K]": air_temp,
        "Process temperature [K]": process_temp,
        "Rotational speed [rpm]": rpm,
        "Torque [Nm]": torque,
        "Tool wear [min]": tool_wear,
    }

    failures = {
        "Machine failure": machine_failure,
        "TWF": twf,
        "HDF": hdf,
        "PWF": pwf,
        "OSF": osf,
        "RNF": rnf,
    }

    return sensors, failures, tool_wear


def generate_full_dataset(n_rows, start_time, id_prefix, udi_offset=0):
    """Generate n_rows of interconnected manufacturing data.
    
    Returns (ai4i_rows, dataset_rows, defect_rows).
    """
    ai4i_rows = []
    dataset_rows = []
    defect_rows = []

    t = start_time
    tool_wear_state = 0
    product_count = 0
    defect_counter = 1

    for i in range(n_rows):
        udi = udi_offset + i + 1
        product_type = sample_product_type()
        product_id = f"{product_type}{udi:05d}"

        line = random.choice(LINES)
        machine_id = f"{line}-M{random.randint(1, MACHINES_PER_LINE):02d}"
        operator_id = random.choice(OPERATORS)
        material_lot = random.choice(MATERIAL_LOTS)

        sensors, failures, tool_wear_state = generate_sensor_row(product_type, tool_wear_state)

        machine_failure = failures["Machine failure"]

        # Determine active failure modes for this row
        active_modes = [m for m in ["TWF", "HDF", "PWF", "OSF", "RNF"] if failures[m]]

        # --- ai4i2020.csv row ---
        ai4i_row = {
            "UDI": udi,
            "Product ID": product_id,
            "Type": product_type,
            **sensors,
            **failures,
            # Extra columns for our system
            "machine_id": machine_id,
            "timestamp": t.isoformat(timespec="seconds"),
            "operator_id": operator_id,
            "material_lot": material_lot,
        }
        ai4i_rows.append(ai4i_row)

        # --- dataset.csv row (machine operation log) ---
        if machine_failure:
            machine_status = random.choices(
                ["Stopped", "Maintenance"], weights=[0.6, 0.4]
            )[0]
        else:
            machine_status = random.choices(
                ["Running", "Idle"], weights=[0.85, 0.15]
            )[0]

        # Quality check: failures don't always get caught at inspection
        # ~75% of actual failures caught, ~2% false alarm on normal
        if machine_failure:
            quality_check = random.choices([False, True], weights=[0.75, 0.25])[0]
        else:
            quality_check = random.choices([True, False], weights=[0.98, 0.02])[0]

        product_count += 1

        dataset_row = {
            "ID": udi,
            "Timestamp": t.isoformat(timespec="seconds"),
            "Machine_ID": machine_id,
            "Air_Temp_K": sensors["Air temperature [K]"],
            "Process_Temp_K": sensors["Process temperature [K]"],
            "RPM": sensors["Rotational speed [rpm]"],
            "Torque_Nm": sensors["Torque [Nm]"],
            "Tool_Wear_Min": sensors["Tool wear [min]"],
            "Machine_Status": machine_status,
            "Quality_Check": quality_check,
            "Product_Count": product_count,
        }
        dataset_rows.append(dataset_row)

        # --- defects_data.csv rows (only for failures) ---
        if machine_failure:
            # Pick the primary failure mode for defect reporting
            primary_mode = active_modes[0] if active_modes else "RNF"
            defect_info = FAILURE_TO_DEFECT[primary_mode]
            sev_weights = SEVERITY_WEIGHTS[primary_mode]

            severity = np.random.choice(
                list(sev_weights.keys()),
                p=list(sev_weights.values()),
            )
            cost_range = REPAIR_COST_RANGE[severity]
            repair_cost = round(random.uniform(*cost_range), 2)

            defect_row = {
                "defect_id": f"DEF-{defect_counter:05d}",
                "product_id": product_id,
                "defect_type": defect_info["defect_type"],
                "defect_date": t.strftime("%Y-%m-%d"),
                "defect_location": random.choice(defect_info["locations"]),
                "severity": severity,
                "inspection_method": random.choice(INSPECTION_METHODS),
                "repair_cost": repair_cost,
                "failure_mode": primary_mode,
                "machine_id": machine_id,
            }
            defect_rows.append(defect_row)
            defect_counter += 1

        # Time increment: 2-6 minutes between batches
        t += timedelta(minutes=random.randint(2, 6))

    return ai4i_rows, dataset_rows, defect_rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs("data", exist_ok=True)

    # 1. Write spec limits
    with open("data/spec_limits.json", "w") as f:
        json.dump(SPEC_LIMITS, f, indent=2)
    print("spec_limits.json written")

    # 2. Generate TRAINING data (what the model trains on)
    print(f"\nGenerating training data ({N_TRAIN} rows)...")
    train_start = datetime(2026, 6, 1, 6, 0, 0)
    ai4i_train, dataset_train, defects_train = generate_full_dataset(
        N_TRAIN, train_start, "HIST", udi_offset=0
    )

    df_ai4i = pd.DataFrame(ai4i_train)
    df_dataset = pd.DataFrame(dataset_train)
    df_defects = pd.DataFrame(defects_train)

    df_ai4i.to_csv("data/ai4i2020.csv", index=False)
    df_dataset.to_csv("data/dataset.csv", index=False)
    df_defects.to_csv("data/defects_data.csv", index=False)

    n_failures = df_ai4i["Machine failure"].sum()
    print(f"  ai4i2020.csv     : {len(df_ai4i)} rows ({n_failures} failures, "
          f"{n_failures / len(df_ai4i) * 100:.1f}% failure rate)")
    print(f"  dataset.csv      : {len(df_dataset)} rows")
    print(f"  defects_data.csv : {len(df_defects)} defect records")

    # Failure mode breakdown
    for mode in ["TWF", "HDF", "PWF", "OSF", "RNF"]:
        count = df_ai4i[mode].sum()
        print(f"    {mode}: {count} ({count / len(df_ai4i) * 100:.2f}%)")

    # 3. Generate LIVE DEMO POOL (unseen by model)
    print(f"\nGenerating live demo pool ({N_POOL} rows)...")
    pool_start = datetime(2026, 7, 29, 6, 0, 0)
    ai4i_pool, dataset_pool, defects_pool = generate_full_dataset(
        N_POOL, pool_start, "BATCH", udi_offset=N_TRAIN
    )

    df_pool = pd.DataFrame(ai4i_pool)
    # Add ground_truth columns for demo validation (model never sees these)
    df_pool["ground_truth_label"] = df_pool["Machine failure"].apply(
        lambda x: "FAILURE" if x == 1 else "NORMAL"
    )
    failure_modes_list = []
    for _, row in df_pool.iterrows():
        modes = [m for m in ["TWF", "HDF", "PWF", "OSF", "RNF"] if row[m] == 1]
        failure_modes_list.append(",".join(modes) if modes else "")
    df_pool["failure_modes"] = failure_modes_list

    df_pool.to_csv("data/log_pool.csv", index=False)

    n_pool_failures = df_pool["Machine failure"].sum()
    print(f"  log_pool.csv     : {len(df_pool)} rows ({n_pool_failures} failures, "
          f"{n_pool_failures / len(df_pool) * 100:.1f}% failure rate) — UNSEEN by model")

    print("\n✓ All data files generated in ./data/")


if __name__ == "__main__":
    main()
