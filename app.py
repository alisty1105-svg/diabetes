from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from modeling import (
    FEATURES,
    TARGET,
    CLASS_NAMES,
    prepare_data,
    clean_dataset,
    detect_outliers_iqr,
    train_and_evaluate,
    predict_patient,
)

# ============================================================
# 1. KONFIGURASI HALAMAN
# ============================================================

st.set_page_config(
    page_title="Diabetes Classification Dashboard",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🩺 Diabetes Classification Dashboard")
st.caption(
    "Dashboard analisis dataset, data cleaning, eksplorasi, "
    "feature engineering, dan klasifikasi diabetes."
)


# ============================================================
# 2. MEMBACA DATASET
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = BASE_DIR / "dataset_diabetes.csv"


@st.cache_data
def load_csv(file_bytes: bytes) -> pd.DataFrame:
    """Membaca CSV dari bytes agar upload dan file lokal sama-sama didukung."""
    from io import BytesIO
    return pd.read_csv(BytesIO(file_bytes))


if not DEFAULT_DATA_PATH.exists():
    st.error(
        "File dataset belum ditemukan. Letakkan "
        "'Multiclass Diabetes Dataset.csv' satu folder dengan app.py, "
        "atau unggah file CSV melalui sidebar."
    )
    uploaded_file = st.sidebar.file_uploader(
        "Unggah Dataset CSV", type=["csv"]
    )
    if uploaded_file is None:
        st.stop()
    df_raw = load_csv(uploaded_file.getvalue())
else:
    with open(DEFAULT_DATA_PATH, "rb") as f:
        local_bytes = f.read()
    uploaded_file = st.sidebar.file_uploader(
        "Ganti Dataset (opsional)", type=["csv"]
    )
    if uploaded_file is not None:
        df_raw = load_csv(uploaded_file.getvalue())
    else:
        df_raw = load_csv(local_bytes)

# Normalisasi dan cleaning dasar untuk analisis/model.
try:
    df_clean = clean_dataset(df_raw, remove_iqr_outliers=False)
except Exception as error:
    st.error(f"Dataset belum dapat diproses: {error}")
    st.stop()

# Simpan sumber data pada session state agar menu-menu memakai data yang sama.
st.session_state["df_raw"] = df_raw
st.session_state["df_clean"] = df_clean

# ============================================================
# 3. MENU NAVIGASI SIDEBAR
# ============================================================

st.sidebar.title("📌 Navigasi Menu")
menu = st.sidebar.radio(
    "Pilih halaman:",
    [
        "Dashboard",
        "Dataset Overview",
        "Data Cleaning and Transformation",
        "Exploratory Data Analysis",
        "Feature Engineering",
        "Data Pasien",
        "Modeling",
        "Hasil Klasifikasi",
        "Prediksi Pasien",
        "Laporan Penelitian",
    ],
)

st.sidebar.markdown("---")
st.sidebar.caption("Proyek klasifikasi diabetes — Multiclass Dataset")
st.sidebar.caption("Target: 0 Non-Diabetic | 1 Diabetic | 2 Prediabetic")

# ============================================================
# 4. FUNGSI BANTU
# ============================================================

def get_numeric_columns(data):
    """Mengambil kolom numerik selain target Class."""
    return [
        col for col in data.select_dtypes(include=np.number).columns
        if col != TARGET
    ]


def show_class_distribution(data, chart_title="Distribusi Kelas"):
    """Menampilkan jumlah dan persentase setiap kelas target."""
    if TARGET not in data.columns:
        st.warning("Kolom Class tidak ditemukan.")
        return

    counts = data[TARGET].value_counts().sort_index()
    labels = [CLASS_NAMES.get(int(x), str(x)) for x in counts.index]
    result = pd.DataFrame({
        "Kelas": labels,
        "Jumlah": counts.values,
        "Persentase (%)": (counts.values / counts.sum() * 100).round(2),
    })

    left, right = st.columns([1, 1])
    with left:
        st.dataframe(result, use_container_width=True, hide_index=True)
    with right:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(result["Kelas"], result["Jumlah"])
        ax.set_title(chart_title)
        ax.set_ylabel("Jumlah data")
        ax.tick_params(axis="x", rotation=15)
        st.pyplot(fig)
        plt.close(fig)


@st.cache_resource
def get_model_result(data_for_model: pd.DataFrame):
    """Melatih model sekali dan menyimpan hasilnya selama sesi/cache."""
    return train_and_evaluate(data_for_model)


def get_model_result_safe(data_for_model):
    try:
        return get_model_result(data_for_model)
    except Exception as error:
        st.error(f"Model belum dapat dilatih: {error}")
        return None

# ============================================================
# 5. HALAMAN DASHBOARD
# ============================================================

if menu == "Dashboard":
    st.header("Dashboard Utama")
    st.write(
        "Halaman ini memberikan ringkasan dataset yang digunakan "
        "untuk mengklasifikasikan kondisi pasien ke dalam tiga kelas."
    )

    total_rows = len(df_raw)
    total_columns = len(df_raw.columns)
    missing_total = int(df_raw.isna().sum().sum())
    duplicate_total = int(df_raw.duplicated().sum())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Jumlah Observasi", f"{total_rows:,}")
    c2.metric("Jumlah Variabel", total_columns)
    c3.metric("Missing Values", missing_total)
    c4.metric("Data Duplikat", duplicate_total)

    st.subheader("Preview Dataset")
    st.dataframe(df_raw.head(10), use_container_width=True)

    st.subheader("Distribusi Target")
    show_class_distribution(df_clean)

    st.info(
        "Mulailah dari Dataset Overview untuk melihat struktur data, "
        "kemudian lanjutkan ke Data Cleaning, EDA, Feature Engineering, "
        "dan Modeling."
    )

# ============================================================
# 6. DATASET OVERVIEW
# ============================================================

elif menu == "Dataset Overview":
    st.header("1. Dataset Overview")
    st.write(
        "Bagian ini menampilkan gambaran awal dataset sebelum proses "
        "pembersihan dan pemodelan."
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Jumlah Baris", f"{df_raw.shape[0]:,}")
    c2.metric("Jumlah Kolom", df_raw.shape[1])
    c3.metric("Jumlah Kelas Target", df_raw[TARGET].nunique() if TARGET in df_raw.columns else 0)

    st.subheader("A. Preview Dataset")
    st.dataframe(df_raw.head(10), use_container_width=True)

    st.subheader("B. Nama dan Tipe Data Variabel")
    info_df = pd.DataFrame({
        "Nama Variabel": df_raw.columns,
        "Tipe Data": [str(x) for x in df_raw.dtypes],
        "Jumlah Data Terisi": [int(df_raw[col].notna().sum()) for col in df_raw.columns],
        "Jumlah Missing": [int(df_raw[col].isna().sum()) for col in df_raw.columns],
    })
    st.dataframe(info_df, use_container_width=True, hide_index=True)

    st.subheader("C. Statistik Deskriptif")
    st.dataframe(df_raw.describe(include="all").T, use_container_width=True)

    st.subheader("D. Distribusi Kelas Target")
    show_class_distribution(df_raw, "Distribusi Class pada Dataset Awal")

    st.subheader("E. Keterangan Variabel")
    descriptions = {
        "Gender": "Jenis kelamin pasien (pada notebook dikodekan secara numerik).",
        "AGE": "Usia pasien.",
        "BMI": "Body Mass Index atau indeks massa tubuh.",
        "Chol": "Kolesterol total.",
        "TG": "Trigliserida.",
        "Urea": "Kadar urea.",
        "VLDL": "Very Low-Density Lipoprotein.",
        "Cr": "Creatinine atau kreatinin.",
        "LDL": "Low-Density Lipoprotein.",
        "HbA1c": "Hemoglobin A1c, indikator rata-rata kadar glukosa darah.",
        "HDL": "High-Density Lipoprotein.",
        "Class": "Target klasifikasi: 0 Non-Diabetic, 1 Diabetic, 2 Prediabetic.",
    }
    desc_table = pd.DataFrame([
        {"Variabel": col, "Keterangan": descriptions.get(col, "Variabel pada dataset.")}
        for col in df_raw.columns
    ])
    st.dataframe(desc_table, use_container_width=True, hide_index=True)


# ============================================================
# 7. DATA CLEANING AND TRANSFORMATION
# ============================================================

elif menu == "Data Cleaning and Transformation":
    st.header("2. Data Cleaning and Transformation")
    st.write(
        "Tahap ini memeriksa data kosong, data duplikat, tipe data, "
        "dan nilai ekstrem (outlier)."
    )

    cleaned = df_clean.copy()
    c1, c2, c3 = st.columns(3)
    c1.metric("Baris Awal", len(df_raw))
    c2.metric("Baris Setelah Cleaning Dasar", len(cleaned))
    c3.metric("Baris yang Tidak Terpakai", len(df_raw) - len(cleaned))

    st.subheader("A. Pemeriksaan Missing Values")
    missing_df = pd.DataFrame({
        "Variabel": df_raw.columns,
        "Jumlah Missing": [int(df_raw[col].isna().sum()) for col in df_raw.columns],
        "Persentase (%)": [
            round(df_raw[col].isna().mean() * 100, 2)
            for col in df_raw.columns
        ],
    })
    st.dataframe(missing_df, use_container_width=True, hide_index=True)

    st.subheader("B. Pemeriksaan Data Duplikat")
    st.write(f"Jumlah baris duplikat: **{int(df_raw.duplicated().sum())}**")

    st.subheader("C. Deteksi Outlier dengan Metode IQR")
    st.write(
        "IQR (Interquartile Range) menggunakan Q1, Q3, dan rentang "
        "1,5 × IQR untuk menandai nilai yang jauh dari sebaran utama."
    )
    outlier_report = detect_outliers_iqr(df_raw)
    st.dataframe(outlier_report, use_container_width=True, hide_index=True)

    outlier_columns = [
        col for col in ["Cr", "VLDL"]
        if col in df_raw.columns and pd.api.types.is_numeric_dtype(df_raw[col])
    ]
    if outlier_columns:
        st.subheader("D. Boxplot Variabel Cr dan VLDL")
        fig, axes = plt.subplots(1, len(outlier_columns), figsize=(6 * len(outlier_columns), 4))
        if len(outlier_columns) == 1:
            axes = [axes]
        for ax, col in zip(axes, outlier_columns):
            sns.boxplot(x=df_raw[col], ax=ax)
            ax.set_title(f"Boxplot {col}")
        st.pyplot(fig)
        plt.close(fig)

    st.subheader("E. Perbandingan Data Sebelum dan Sesudah Cleaning Dasar")
    compare = pd.DataFrame({
        "Keterangan": ["Jumlah baris", "Jumlah kolom", "Data duplikat"],
        "Sebelum": [
            len(df_raw), len(df_raw.columns), int(df_raw.duplicated().sum())
        ],
        "Sesudah": [
            len(cleaned), len(cleaned.columns), int(cleaned.duplicated().sum())
        ],
    })
    st.dataframe(compare, use_container_width=True, hide_index=True)

    st.subheader("F. Opsi Penghapusan Outlier Cr dan VLDL")
    st.warning(
        "Penghapusan outlier tidak dilakukan otomatis. Notebook memuat "
        "contoh penghapusan outlier Cr/VLDL, tetapi bagian interpretasinya "
        "juga menyatakan outlier dapat valid secara klinis. Pastikan satuan "
        "dan validitas nilai diperiksa sebelum menghapusnya."
    )
    remove_outliers = st.checkbox(
        "Saya sudah memverifikasi dan ingin menghapus outlier IQR pada Cr dan VLDL",
        value=False,
    )
    if remove_outliers:
        cleaned_optional = clean_dataset(df_raw, remove_iqr_outliers=True)
        st.write(
            f"Jumlah baris setelah penghapusan outlier opsional: "
            f"**{len(cleaned_optional)}**"
        )
        st.dataframe(cleaned_optional.head(20), use_container_width=True)
        st.download_button(
            "Unduh Dataset Setelah Penghapusan Outlier",
            data=cleaned_optional.to_csv(index=False).encode("utf-8"),
            file_name="diabetes_dataset_cleaned_iqr.csv",
            mime="text/csv",
        )
    else:
        st.success(
            "Dataset model menggunakan cleaning dasar: konversi numerik, "
            "penghapusan duplikat, dan penghapusan baris yang tidak lengkap "
            "pada fitur model/target. Outlier tetap dipertahankan."
        )

    st.download_button(
        "Unduh Dataset Hasil Cleaning Dasar",
        data=cleaned.to_csv(index=False).encode("utf-8"),
        file_name="diabetes_dataset_cleaned.csv",
        mime="text/csv",
    )

# ============================================================
# 8. EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================

elif menu == "Exploratory Data Analysis":
    st.header("3. Exploratory Data Analysis (EDA)")
    st.write(
        "EDA digunakan untuk memahami pola distribusi setiap variabel "
        "dan hubungan variabel dengan kelas diabetes."
    )

    numeric_cols = get_numeric_columns(df_clean)

    st.subheader("A. Univariate Analysis — Distribusi Setiap Variabel")
    selected_univariate = st.selectbox(
        "Pilih variabel numerik:",
        numeric_cols,
    )
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(df_clean[selected_univariate], kde=True, bins=20, ax=ax)
    ax.set_title(f"Distribusi {selected_univariate}")
    st.pyplot(fig)
    plt.close(fig)

    if numeric_cols:
        if st.checkbox("Tampilkan histogram seluruh variabel numerik"):
            ncols = 3
            nrows = int(np.ceil(len(numeric_cols) / ncols))
            fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4 * nrows))
            axes = np.array(axes).reshape(-1)
            for ax, col in zip(axes, numeric_cols):
                sns.histplot(df_clean[col], kde=True, bins=20, ax=ax)
                ax.set_title(col)
            for ax in axes[len(numeric_cols):]:
                ax.axis("off")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

    st.subheader("B. Bivariate Analysis — Fitur terhadap Class")
    feature_options = [col for col in numeric_cols if col != TARGET]
    selected_feature = st.selectbox(
        "Pilih fitur untuk dibandingkan dengan Class:",
        feature_options,
        key="eda_bivariate_feature",
    )
    plot_df = df_clean.copy()
    plot_df["Nama Kelas"] = plot_df[TARGET].map(CLASS_NAMES)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(
        data=plot_df,
        x="Nama Kelas",
        y=selected_feature,
        ax=ax,
    )
    ax.set_title(f"{selected_feature} berdasarkan kelas diabetes")
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("C. Analisis Gender terhadap Class")
    if "Gender" in df_clean.columns:
        gender_class = pd.crosstab(df_clean["Gender"], df_clean[TARGET])
        gender_class.columns = [
            CLASS_NAMES.get(int(col), str(col)) for col in gender_class.columns
        ]
        st.dataframe(gender_class, use_container_width=True)

        fig, ax = plt.subplots(figsize=(7, 4))
        gender_class.plot(kind="bar", ax=ax)
        ax.set_title("Distribusi kelas berdasarkan Gender")
        ax.set_xlabel("Kode Gender")
        ax.set_ylabel("Jumlah")
        ax.tick_params(axis="x", rotation=0)
        st.pyplot(fig)
        plt.close(fig)

    st.subheader("D. Distribusi Target")
    show_class_distribution(df_clean, "Distribusi Kelas Diabetes")

# ============================================================
# 9. FEATURE ENGINEERING
# ============================================================

elif menu == "Feature Engineering":
    st.header("4. Feature Engineering")
    st.write(
        "Tahap ini mempelajari hubungan antarvariabel dan menampilkan "
        "hasil seleksi fitur menggunakan RFE."
    )

    feature_numeric = get_numeric_columns(df_clean)
    corr_data = df_clean[feature_numeric].corr(numeric_only=True)

    st.subheader("A. Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(corr_data, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    ax.set_title("Korelasi Antarvariabel Numerik")
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("B. Korelasi Fitur terhadap Class")
    if TARGET in corr_data.columns:
        target_corr = corr_data[TARGET].drop(TARGET).sort_values(
            key=lambda x: x.abs(), ascending=False
        )
        st.dataframe(
            target_corr.rename("Korelasi dengan Class").to_frame(),
            use_container_width=True,
        )

    st.subheader("C. Recursive Feature Elimination (RFE)")
    st.write(
        "RFE melatih estimator secara berulang untuk menentukan peringkat "
        "fitur. Untuk menghindari data leakage, RFE di bawah ini dipasang "
        "hanya pada bagian training."
    )

    try:
        from sklearn.feature_selection import RFE
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import train_test_split

        X_all = df_clean.drop(columns=[TARGET]).select_dtypes(include=np.number)
        y_all = df_clean[TARGET].astype(int)
        X_train_rfe, _, y_train_rfe, _ = train_test_split(
            X_all,
            y_all,
            test_size=0.20,
            stratify=y_all,
            random_state=42,
        )

        scaler_rfe = StandardScaler()
        X_train_scaled_rfe = scaler_rfe.fit_transform(X_train_rfe)

        estimator = LogisticRegression(max_iter=2000, class_weight="balanced")
        rfe = RFE(estimator=estimator, n_features_to_select=min(5, X_all.shape[1]))
        rfe.fit(X_train_scaled_rfe, y_train_rfe)

        rfe_results = pd.DataFrame({
            "Fitur": X_train_rfe.columns,
            "Ranking": rfe.ranking_,
            "Terpilih": rfe.support_,
        }).sort_values(["Ranking", "Fitur"])
        st.dataframe(rfe_results, use_container_width=True, hide_index=True)

        selected_rfe = rfe_results.loc[
            rfe_results["Terpilih"], "Fitur"
        ].tolist()
        st.write("Fitur terpilih dari RFE pada data training:", selected_rfe)
    except Exception as error:
        st.warning(f"RFE belum dapat dijalankan: {error}")

    st.info(
        "Model pada notebook menggunakan lima fitur final: "
        + ", ".join(FEATURES)
        + ". RFE di halaman ini merupakan pemeriksaan/eksplorasi pada data "
          "training; daftar fitur yang tampil dapat berbeda karena pembagian "
          "data dan proses seleksi."
    )


# ============================================================
# 10. DATA PASIEN
# ============================================================

elif menu == "Data Pasien":
    st.header("5. Data Pasien")
    st.write("Tabel data pasien yang telah melalui cleaning dasar.")
    st.metric("Jumlah Data Pasien Valid", len(df_clean))
    st.dataframe(df_clean, use_container_width=True)

    st.download_button(
        "Unduh Data Pasien",
        data=df_clean.to_csv(index=False).encode("utf-8"),
        file_name="data_pasien_diabetes.csv",
        mime="text/csv",
    )

# ============================================================
# 11. MODELING
# ============================================================

elif menu == "Modeling":
    st.header("6. Modeling")
    st.write(
        "Tiga algoritma dibandingkan: Logistic Regression, Random Forest, "
        "dan Gradient Boosting. Pembagian data menggunakan 80% training "
        "dan 20% testing secara stratified."
    )
    st.write("Fitur model:", ", ".join(FEATURES))

    with st.spinner("Melatih dan mengevaluasi model..."):
        result = get_model_result_safe(df_clean)

    if result is not None:
        st.success(f"Model dengan Macro Recall tertinggi: {result['best_model_name']}")
        st.subheader("Perbandingan Kinerja Model")
        st.dataframe(
            result["metrics"].style.format("{:.4f}"),
            use_container_width=True,
        )

        st.subheader("Cross-Validation Random Forest")
        scores = result["cv_scores"]
        if len(scores):
            cv_table = pd.DataFrame({
                "Fold": [f"Fold {i}" for i in range(1, len(scores) + 1)],
                "Macro F1": scores,
            })
            st.dataframe(cv_table, use_container_width=True, hide_index=True)
            a, b = st.columns(2)
            a.metric("Rata-rata Macro F1", f"{scores.mean():.4f}")
            b.metric("Standar Deviasi", f"{scores.std():.4f}")
        else:
            st.warning("Cross-validation tidak dapat dijalankan untuk jumlah data per kelas saat ini.")

        st.caption(
            "Catatan: Gradient Boosting pada scikit-learn tidak memiliki "
            "parameter class_weight. Logistic Regression dan Random Forest "
            "menggunakan class_weight='balanced'."
        )

# ============================================================
# 12. HASIL KLASIFIKASI
# ============================================================

elif menu == "Hasil Klasifikasi":
    st.header("7. Hasil Klasifikasi dan Evaluasi")
    with st.spinner("Menyiapkan hasil pengujian model..."):
        result = get_model_result_safe(df_clean)

    if result is not None:
        model_names = list(result["models"].keys())
        chosen_model_name = st.selectbox(
            "Pilih model untuk melihat hasil evaluasi:",
            model_names,
            index=model_names.index(result["best_model_name"]),
        )
        detail = result["details"][chosen_model_name]

        st.subheader(f"Confusion Matrix — {chosen_model_name}")
        cm = detail["confusion_matrix"]
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=[CLASS_NAMES[i] for i in [0, 1, 2]],
            yticklabels=[CLASS_NAMES[i] for i in [0, 1, 2]],
            ax=ax,
        )
        ax.set_xlabel("Prediksi")
        ax.set_ylabel("Aktual")
        st.pyplot(fig)
        plt.close(fig)

        st.subheader("Classification Report")
        report_df = pd.DataFrame(detail["classification_report"]).T
        st.dataframe(report_df, use_container_width=True)

        st.subheader("Ringkasan Metrik")
        st.dataframe(
            result["metrics"].loc[[chosen_model_name]].style.format("{:.4f}"),
            use_container_width=True,
        )

        st.subheader("Prediksi pada Data Testing")
        st.dataframe(result["test_predictions"], use_container_width=True)

        st.download_button(
            "Unduh Hasil Prediksi Data Testing",
            data=result["test_predictions"].to_csv(index=False).encode("utf-8"),
            file_name="hasil_klasifikasi_data_testing.csv",
            mime="text/csv",
        )

# ============================================================
# 13. PREDIKSI PASIEN BARU
# ============================================================

elif menu == "Prediksi Pasien":
    st.header("8. Prediksi Pasien Baru")
    st.write(
        "Masukkan nilai lima fitur yang digunakan model. Hasil ini merupakan "
        "prediksi model, bukan diagnosis medis."
    )

    with st.spinner("Memuat model terbaik..."):
        result = get_model_result_safe(df_clean)

    if result is not None:
        best_model = result["best_model"]
        st.caption(f"Model yang digunakan: {result['best_model_name']}")

        medians = df_clean[FEATURES].median(numeric_only=True)

        with st.form("patient_prediction_form"):
            col1, col2 = st.columns(2)

            with col1:
                gender = st.selectbox(
                    "Gender",
                    options=[0, 1],
                    format_func=lambda x: (
                        "0 — Perempuan" if x == 0 else "1 — Laki-laki"
                    ),
                )
                hba1c = st.number_input(
                    "HbA1c",
                    min_value=0.0,
                    max_value=30.0,
                    value=float(medians["HbA1c"]),
                    step=0.1,
                )
                tg = st.number_input(
                    "TG (Trigliserida)",
                    min_value=0.0,
                    max_value=100.0,
                    value=float(medians["TG"]),
                    step=0.1,
                )

            with col2:
                vldl = st.number_input(
                    "VLDL",
                    min_value=0.0,
                    max_value=100.0,
                    value=float(medians["VLDL"]),
                    step=0.1,
                )
                bmi = st.number_input(
                    "BMI",
                    min_value=0.0,
                    max_value=100.0,
                    value=float(medians["BMI"]),
                    step=0.1,
                )

            submitted = st.form_submit_button("Prediksi Kelas Diabetes")

        if submitted:
            patient = {
                "Gender": float(gender),
                "HbA1c": float(hba1c),
                "TG": float(tg),
                "VLDL": float(vldl),
                "BMI": float(bmi),
            }
            prediction = predict_patient(best_model, patient)

            st.subheader("Hasil Prediksi")
            st.success(
                f"Prediksi kelas: {prediction['class_id']} — "
                f"{prediction['class_name']}"
            )

            probability_df = pd.DataFrame({
                "Kelas": list(prediction["probabilities"].keys()),
                "Probabilitas (%)": [
                    round(value * 100, 2)
                    for value in prediction["probabilities"].values()
                ],
            })
            st.dataframe(probability_df, use_container_width=True, hide_index=True)

            fig, ax = plt.subplots(figsize=(7, 4))
            ax.bar(
                probability_df["Kelas"],
                probability_df["Probabilitas (%)"],
            )
            ax.set_ylabel("Probabilitas (%)")
            ax.set_ylim(0, 100)
            ax.set_title("Probabilitas Prediksi Setiap Kelas")
            st.pyplot(fig)
            plt.close(fig)

            st.warning(
                "Hasil ini hanya alat bantu pembelajaran/analisis data. "
                "Interpretasi dan keputusan klinis harus dilakukan oleh "
                "tenaga kesehatan berdasarkan pemeriksaan yang sesuai."
            )


# ============================================================
# 14. LAPORAN PENELITIAN
# ============================================================

elif menu == "Laporan Penelitian":
    st.header("9. Ringkasan Laporan Penelitian")
    st.write("Ringkasan tahapan analisis yang tersedia pada dashboard.")

    stages = [
        ("Business/Data Understanding", "Memahami tujuan klasifikasi dan struktur dataset."),
        ("Dataset Overview", "Melihat jumlah observasi, variabel, tipe data, dan target."),
        ("Data Cleaning", "Memeriksa missing values, duplikasi, tipe data, dan outlier."),
        ("EDA", "Menganalisis distribusi variabel dan hubungan fitur terhadap Class."),
        ("Feature Engineering", "Menganalisis korelasi dan seleksi fitur menggunakan RFE."),
        ("Modeling", "Membandingkan Logistic Regression, Random Forest, dan Gradient Boosting."),
        ("Evaluation", "Mengevaluasi Accuracy, Macro Precision, Macro Recall, Macro F1, dan ROC-AUC."),
    ]
    for stage, explanation in stages:
        st.markdown(f"**{stage}** — {explanation}")

    st.subheader("Catatan Metodologis")
    st.markdown(
        """
        - Dataset pada notebook dijelaskan memiliki 264 observasi dan 12 variabel.
        - Target terdiri dari tiga kelas: 0 Non-Diabetic, 1 Diabetic, dan 2 Prediabetic.
        - Lima fitur model mengikuti daftar pada notebook: Gender, HbA1c, TG, VLDL, dan BMI.
        - Data dibagi menjadi training dan testing dengan rasio 80:20 serta stratifikasi.
        - Standardisasi diterapkan pada Logistic Regression melalui pipeline; Random Forest
          dan Gradient Boosting tidak memerlukan standardisasi fitur untuk proses dasarnya.
        - Nilai performa yang tampil di dashboard dihitung ulang dari file CSV yang digunakan,
          sehingga dapat berbeda dari angka narasi pada notebook.
        """
    )

st.sidebar.markdown("---")
st.sidebar.caption("Catatan: hasil klasifikasi bukan pengganti diagnosis medis.")
