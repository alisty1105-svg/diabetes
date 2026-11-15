from typing import Dict, Any

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)

TARGET = "Class"
FEATURES = ["Gender", "HbA1c", "TG", "VLDL", "BMI"]

CLASS_NAMES = {
    0: "Non-Diabetic",
    1: "Diabetic",
    2: "Prediabetic",
}


def normalize_column_names(data: pd.DataFrame) -> pd.DataFrame:
    """Rapikan nama kolom dan samakan variasi Age/AGE menjadi AGE."""
    df = data.copy()
    df.columns = [str(col).strip() for col in df.columns]

    rename_map = {}
    for col in df.columns:
        if col.lower() == "age":
            rename_map[col] = "AGE"
    if rename_map:
        df = df.rename(columns=rename_map)

    return df

def prepare_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Persiapan data dasar:
    - menormalkan nama kolom;
    - mengubah kolom menjadi numerik jika memungkinkan;
    - menghapus duplikasi;
    - menghapus baris yang tidak memiliki fitur model atau target lengkap.
    """
    df = normalize_column_names(data)

    required = FEATURES + [TARGET]
    missing_columns = [col for col in required if col not in df.columns]
    if missing_columns:
        raise ValueError(
            "Kolom wajib tidak ditemukan: " + ", ".join(missing_columns)
        )

    # Dataset pada notebook menggunakan Gender dalam bentuk angka.
    # Jika Gender berupa teks, konversi nilai umum ke kode 0/1.
    if df["Gender"].dtype == "object":
        gender_map = {
            "female": 0, "f": 0, "perempuan": 0, "wanita": 0,
            "male": 1, "m": 1, "laki-laki": 1, "pria": 1,
        }
        df["Gender"] = (
            df["Gender"].astype(str).str.strip().str.lower().map(gender_map)
            .where(
                df["Gender"].astype(str).str.strip().str.lower().isin(gender_map),
                df["Gender"]
            )
        )

    for col in required:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.drop_duplicates()
    df = df.dropna(subset=required)

    # Hanya kelas 0, 1, dan 2 yang digunakan.
    df = df[df[TARGET].isin([0, 1, 2])].copy()
    df[TARGET] = df[TARGET].astype(int)

    if df.empty:
        raise ValueError(
            "Tidak ada baris data valid. Periksa kolom fitur dan kolom Class."
        )

    return df.reset_index(drop=True)

def detect_outliers_iqr(data: pd.DataFrame, columns=None) -> pd.DataFrame:
    """Menghasilkan ringkasan jumlah outlier berdasarkan aturan IQR."""
    df = normalize_column_names(data)
    if columns is None:
        columns = [
            col for col in df.select_dtypes(include=np.number).columns
            if col != TARGET
        ]

    rows = []
    for col in columns:
        if col not in df.columns:
            continue
        values = pd.to_numeric(df[col], errors="coerce").dropna()
        if values.empty:
            continue

        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        count = int(((values < lower) | (values > upper)).sum())

        rows.append({
            "Variabel": col,
            "Jumlah Outlier": count,
            "Persentase (%)": round(count / len(values) * 100, 2),
            "Batas Bawah IQR": round(float(lower), 4),
            "Batas Atas IQR": round(float(upper), 4),
        })

    return pd.DataFrame(rows)


def clean_dataset(data: pd.DataFrame, remove_iqr_outliers: bool = False) -> pd.DataFrame:
    """
    Membersihkan data dasar.
    Penghapusan outlier Cr dan VLDL bersifat opsional, bukan default.
    Aktifkan hanya jika sudah diverifikasi sebagai kesalahan/tidak valid.
    """
    df = prepare_data(data)

    if remove_iqr_outliers:
        for col in ["Cr", "VLDL"]:
            if col not in df.columns:
                continue
            values = pd.to_numeric(df[col], errors="coerce")
            q1 = values.quantile(0.25)
            q3 = values.quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            df = df[(values >= lower) & (values <= upper)].copy()

    return df.reset_index(drop=True)

def build_models(random_state: int = 42) -> Dict[str, Any]:
    """Membuat tiga model yang digunakan pada notebook."""
    return {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                max_iter=2000,
                solver="lbfgs",
                class_weight="balanced",
                random_state=random_state,
            )),
        ]),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=random_state,
        ),
        # GradientBoostingClassifier tidak menyediakan parameter class_weight.
        "Gradient Boosting": GradientBoostingClassifier(
            random_state=random_state
        ),
    }

def train_and_evaluate(
    data: pd.DataFrame,
    test_size: float = 0.20,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Membagi data secara stratified 80:20, melatih tiga model,
    menghitung metrik pengujian, dan cross-validation Random Forest.
    """
    df = prepare_data(data)

    absent = [col for col in FEATURES + [TARGET] if col not in df.columns]
    if absent:
        raise ValueError("Kolom tidak tersedia: " + ", ".join(absent))

    X = df[FEATURES].astype(float)
    y = df[TARGET].astype(int)

    class_counts = y.value_counts()
    if len(class_counts) < 2 or class_counts.min() < 2:
        raise ValueError(
            "Setiap kelas harus memiliki minimal 2 data agar pembagian "
            "train-test stratified dapat dilakukan."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    models = build_models(random_state)
    fitted_models = {}
    metric_rows = []
    details = {}

    for model_name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)

        try:
            auc = roc_auc_score(
                y_test,
                probabilities,
                multi_class="ovr",
                average="macro",
                labels=model.classes_,
            )
        except ValueError:
            auc = np.nan

        metric_rows.append({
            "Model": model_name,
            "Accuracy": accuracy_score(y_test, predictions),
            "Macro Precision": precision_score(
                y_test, predictions, average="macro", zero_division=0
            ),
            "Macro Recall": recall_score(
                y_test, predictions, average="macro", zero_division=0
            ),
            "Macro F1": f1_score(
                y_test, predictions, average="macro", zero_division=0
            ),
            "Macro ROC-AUC": auc,
        })

        fitted_models[model_name] = model
        details[model_name] = {
            "predictions": predictions,
            "probabilities": probabilities,
            "confusion_matrix": confusion_matrix(
                y_test, predictions, labels=[0, 1, 2]
            ),
            "classification_report": classification_report(
                y_test,
                predictions,
                labels=[0, 1, 2],
                target_names=[
                    CLASS_NAMES[0], CLASS_NAMES[1], CLASS_NAMES[2]
                ],
                output_dict=True,
                zero_division=0,
            ),
        }

    metrics = pd.DataFrame(metric_rows).set_index("Model")
    # Pemilihan otomatis berdasarkan Macro Recall, sesuai prioritas notebook.
    best_model_name = metrics["Macro Recall"].idxmax()
    best_model = fitted_models[best_model_name]

    # CV dilakukan pada data training saja.
    min_class_count = int(y_train.value_counts().min())
    n_splits = min(5, min_class_count)
    cv_scores = np.array([])
    if n_splits >= 2:
        cv = StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=random_state
        )
        cv_scores = cross_val_score(
            fitted_models["Random Forest"],
            X_train,
            y_train,
            cv=cv,
            scoring="f1_macro",
        )

    test_predictions = pd.DataFrame({
        "Index Data Uji": X_test.index,
        "Class Aktual": y_test.values,
        "Prediksi Class": details[best_model_name]["predictions"],
    })
    test_predictions["Kelas Aktual"] = test_predictions["Class Aktual"].map(CLASS_NAMES)
    test_predictions["Nama Prediksi"] = test_predictions["Prediksi Class"].map(CLASS_NAMES)

    return {
        "data": df,
        "features": FEATURES,
        "models": fitted_models,
        "metrics": metrics.sort_values("Macro Recall", ascending=False),
        "best_model_name": best_model_name,
        "best_model": best_model,
        "details": details,
        "X_test": X_test,
        "y_test": y_test,
        "test_predictions": test_predictions,
        "cv_scores": cv_scores,
        "train_size": len(X_train),
        "test_size": len(X_test),
    }


def train_models(data: pd.DataFrame) -> Dict[str, Any]:
    """Alias agar pemanggilan train_models(data) tetap mudah digunakan."""
    return train_and_evaluate(data)


def predict_patient(model, patient_data: Dict[str, float]) -> Dict[str, Any]:
    """Memprediksi satu pasien dan mengembalikan kelas serta probabilitas."""
    patient_df = pd.DataFrame([patient_data], columns=FEATURES)
    prediction = int(model.predict(patient_df)[0])

    probabilities = model.predict_proba(patient_df)[0]
    class_order = list(model.classes_)

    probability_result = {
        CLASS_NAMES[int(class_id)]: float(probability)
        for class_id, probability in zip(class_order, probabilities)
    }

    return {
        "class_id": prediction,
    }