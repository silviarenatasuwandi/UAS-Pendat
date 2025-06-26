from ucimlrepo import fetch_ucirepo
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib

data = fetch_ucirepo(id=519)
X = data.data.features
y = data.data.targets.values.ravel()

X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

joblib.dump(model, "heart_failure_model.pkl")
joblib.dump(scaler, "heart_failure_scaler.pkl")

print("✅ Model dan scaler berhasil disimpan.")
