import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder

# Load datasets with optimized memory usage
patients = pd.read_csv("PATIENTS(c).csv", low_memory=False)
icustays = pd.read_csv("ICUSTAYS(c).csv", low_memory=False)
admissions = pd.read_csv("ADMISSIONS(c).csv", low_memory=False)
chartevents = pd.read_csv("CHARTEVENTS(c).csv", low_memory=False, 
                          dtype={"value": str, "warning": str, "error": str, "resultstatus": str, "stopped": str})
labevents = pd.read_csv("LABEVENTS(c).csv", low_memory=False)

# Fix ICU admissions count per patient
icu_counts = icustays.groupby("subject_id")["icustay_id"].count().reset_index()
icu_counts.columns = ["subject_id", "total_icu_admissions"]
patients = patients.merge(icu_counts, on="subject_id", how="left")
patients["total_icu_admissions"] = patients["total_icu_admissions"].fillna(0)

# Fix datetime parsing
date_cols = ["intime", "outtime"]
icustays[date_cols] = icustays[date_cols].apply(pd.to_datetime, errors="coerce")

icustays["icu_los_hours"] = (icustays["outtime"] - icustays["intime"]).dt.total_seconds() / 3600

admissions["admittime"] = pd.to_datetime(admissions["admittime"], errors="coerce")
admissions["dischtime"] = pd.to_datetime(admissions["dischtime"], errors="coerce")
admissions["hospital_los_days"] = (admissions["dischtime"] - admissions["admittime"]).dt.days

# Merge ICU stay info with admissions
admissions = admissions.merge(icustays[["hadm_id", "icu_los_hours"]], on="hadm_id", how="left")

# Feature engineering for vitals from chartevents
chartevents["charttime"] = pd.to_datetime(chartevents["charttime"], errors="coerce")
labevents["charttime"] = pd.to_datetime(labevents["charttime"], errors="coerce")

vitals = chartevents.groupby(["subject_id", "hadm_id", "icustay_id"]) \
    .agg(mean_vital=("valuenum", "mean"),
         max_vital=("valuenum", "max"),
         min_vital=("valuenum", "min")).reset_index()

# One-hot encoding for categorical variables
categorical_cols = ["insurance", "language", "religion", "marital_status", "ethnicity"]
admissions[categorical_cols] = admissions[categorical_cols].fillna("Unknown")  # Handle missing values

encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
encoded_features = encoder.fit_transform(admissions[categorical_cols])
encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(categorical_cols))
admissions = pd.concat([admissions.drop(columns=categorical_cols), encoded_df], axis=1)

# Privacy-aware feature engineering (Differential Privacy)
def add_laplace_noise(data, epsilon=1.0):
    noise = np.random.laplace(0, 1/epsilon, size=len(data))
    return data + noise

if "age" in patients.columns:
    patients["age_noisy"] = add_laplace_noise(patients["age"])

# Save processed datasets
patients.to_csv("patients_processed.csv", index=False)
admissions.to_csv("admissions_processed.csv", index=False)
icustays.to_csv("icustays_processed.csv", index=False)
vitals.to_csv("vitals_processed.csv", index=False)

print("Feature engineering complete! Processed datasets saved.")
