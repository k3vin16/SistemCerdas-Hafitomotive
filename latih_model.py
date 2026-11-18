import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

fitur_kategorikal = ["merek", "model", "transmisi", "bahan_bakar", "status_pajak", "kota"]
fitur_numerik = ["tahun", "kilometer", "kapasitas_cc"]
semua_fitur = fitur_kategorikal + fitur_numerik


def latih_satu_kategori(nama_file, label):
    print(f"\n--- Memproses Dataset {label.upper()} ({nama_file}) ---")
    df = pd.read_csv(nama_file)

    # Bersihkan spasi berlebih pada kolom teks
    for kol in fitur_kategorikal:
        df[kol] = df[kol].fillna("Lainnya").astype(str).str.strip()

    # Isi nilai kosong pada fitur numerik dengan median agar seluruh baris (terutama motor) ikut terlatih
    for kol in fitur_numerik:
        df[kol] = pd.to_numeric(df[kol], errors="coerce")
        df[kol] = df[kol].fillna(df[kol].median())

    df = df.dropna(subset=["harga"]).copy()

    X = df[semua_fitur]
    y = df["harga"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("kat", OneHotEncoder(handle_unknown="ignore"), fitur_kategorikal),
            ("num", "passthrough", fitur_numerik),
        ]
    )

    pipeline_model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", RandomForestRegressor(n_estimators=150, random_state=42)),
        ]
    )

    pipeline_model.fit(X_train, y_train)
    y_pred = pipeline_model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Evaluasi {label.upper()} ({len(df)} data) -> R2 Score: {r2:.3f} | MAE: Rp {mae:,.0f}")

    rata_harga_merek = (
        df.groupby("merek")["harga"]
        .mean()
        .sort_values(ascending=False)
        .head(6)
        .round(0)
        .to_dict()
    )

    merek_ke_model = (
        df.groupby("merek")["model"]
        .apply(lambda s: sorted(s.unique().tolist()))
        .to_dict()
    )

    daftar_opsi = {
        "merek": sorted(df["merek"].unique().tolist()),
        "model": sorted(df["model"].unique().tolist()),
        "transmisi": sorted(df["transmisi"].unique().tolist()),
        "bahan_bakar": sorted(df["bahan_bakar"].unique().tolist()),
        "merek_ke_model": merek_ke_model,
    }

    return {
        "model": pipeline_model,
        "opsi_dropdown": daftar_opsi,
        "grafik_merek": rata_harga_merek,
        "metrik": {"r2": round(r2, 3), "mae": int(mae)},
    }


hasil_mobil = latih_satu_kategori("dataset_olx_mobil_real.csv", "mobil")
hasil_motor = latih_satu_kategori("dataset_olx_motor_real.csv", "motor")

artefak_gabungan = {
    "fitur_kolom": semua_fitur,
    "mobil": hasil_mobil,
    "motor": hasil_motor,
}

joblib.dump(artefak_gabungan, "model.joblib")
print("\nSUKSES! Model Mobil & Motor tersimpan ke 'model.joblib'.")