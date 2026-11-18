import os
import json
import joblib
import numpy as np
import urllib.request
import urllib.error
import pandas as pd
from urllib.parse import quote_plus
from django.conf import settings
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.http import require_GET

# ================= KONFIGURASI GROQ API =================
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "").strip()
GROQ_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODELS_ENDPOINT = "https://api.groq.com/openai/v1/models"

MODEL_PATH = os.path.join(settings.BASE_DIR, "model.joblib")
ARTEFAK = joblib.load(MODEL_PATH)
FITUR_KOLOM = ARTEFAK["fitur_kolom"]

CACHE_FOTO_OLX = {}
CACHE_ANALISIS_GRAFIK = {}
CACHE_MODEL_AKTIF = None


def dapatkan_model_groq_aktif():
    """Mengambil otomatis ID model chat yang aktif dari akun Groq Anda."""
    global CACHE_MODEL_AKTIF
    if CACHE_MODEL_AKTIF:
        return CACHE_MODEL_AKTIF

    req = urllib.request.Request(
        GROQ_MODELS_ENDPOINT,
        headers={
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "User-Agent": "Mozilla/5.0",
        },
        method="GET",
    )
    try:
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            semua_id = [
                m["id"] for m in data.get("data", [])
                if "whisper" not in m["id"].lower()
                and "guard" not in m["id"].lower()
                and "tts" not in m["id"].lower()
            ]
            prioritas = [m for m in semua_id if any(k in m.lower() for k in ["llama", "qwen", "gemma", "deepseek"])]
            daftar_final = prioritas if prioritas else semua_id
            if daftar_final:
                CACHE_MODEL_AKTIF = daftar_final[:3]
                return CACHE_MODEL_AKTIF
    except Exception:
        pass

    return ["llama-3.1-8b-instant", "llama-3.3-70b-versatile", "openai/gpt-oss-20b"]


def tanya_groq_ai(system_prompt, user_prompt, max_tokens=500):
    """Memanggil Groq API menggunakan model aktif yang terdeteksi otomatis."""
    if not GROQ_API_KEY or "MASUKKAN_API_KEY" in GROQ_API_KEY:
        return None

    daftar_model = dapatkan_model_groq_aktif()
    daftar_error = []

    for nama_model in daftar_model:
        payload = {
            "model": nama_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.4,
            "max_tokens": max_tokens,
        }

        req = urllib.request.Request(
            GROQ_ENDPOINT,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json",
                "User-Agent": "Mozilla/5.0",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=12) as resp:
                hasil = json.loads(resp.read().decode("utf-8"))
                return hasil["choices"][0]["message"]["content"].strip()
        except urllib.error.HTTPError as he:
            detail = he.read().decode("utf-8", errors="ignore")
            daftar_error.append(f"[{nama_model}] HTTP {he.code}: {detail}")
            if he.code in [401, 403]:
                break
        except Exception as e:
            daftar_error.append(f"[{nama_model}] {e}")

    return "Gagal memuat analisis AI: " + " | ".join(daftar_error)


def muat_dataset_bersih(nama_csv):
    path_csv = os.path.join(settings.BASE_DIR, nama_csv)
    if not os.path.exists(path_csv):
        return pd.DataFrame()
    df = pd.read_csv(path_csv)
    for kol in ["merek", "model", "varian", "transmisi", "bahan_bakar", "status_pajak", "kota", "id_listing"]:
        if kol in df.columns:
            df[kol] = df[kol].fillna("Lainnya").astype(str).str.strip()
    for kol in ["tahun", "kilometer", "kapasitas_cc", "harga"]:
        if kol in df.columns:
            df[kol] = pd.to_numeric(df[kol], errors="coerce")
    return df.dropna(subset=["harga"]).copy()


DATASET_REKOMENDASI = {
    "mobil": muat_dataset_bersih("dataset_olx_mobil_real.csv"),
    "motor": muat_dataset_bersih("dataset_olx_motor_real.csv"),
}


def bangun_url_olx(id_listing, merek, model):
    angka_id = str(id_listing).replace("OLX-", "").strip()
    url_item = f"https://www.olx.co.id/item/iid-{angka_id}"
    query_cari = quote_plus(f"{merek} {model}")
    url_cari = f"https://www.olx.co.id/jawa-timur_g2000009/q-{query_cari}"
    return url_item, url_cari


def format_kartu_rekomendasi(df_pilihan, harga_acuan):
    """Menampilkan seluruh pilihan unit untuk Merek & Model terpilih (lintas transmisi & bahan bakar)."""
    hasil = []
    if df_pilihan.empty:
        return hasil

    df_rek = df_pilihan.copy()
    df_rek["selisih"] = (df_rek["harga"] - harga_acuan).abs()
    # Tampilkan hingga 12 unit terbaik agar mencakup variasi AT, MT, Bensin, Diesel, dll.
    df_rek = df_rek.sort_values(by=["selisih", "harga"]).head(12)

    for _, row in df_rek.iterrows():
        harga_unit = float(row["harga"])
        hemat = harga_unit <= harga_acuan
        id_list = row.get("id_listing", "OLX-000000")
        url_item, url_cari = bangun_url_olx(id_list, row["merek"], row["model"])
        tahun_str = str(int(row["tahun"])) if pd.notna(row["tahun"]) else "-"
        km_str = f"{int(row['kilometer']):,} km".replace(",", ".") if pd.notna(row["kilometer"]) else "KM tidak dicantumkan"
        cc_str = f"{int(row['kapasitas_cc'])} cc" if pd.notna(row["kapasitas_cc"]) else "-"
        varian_val = row.get("varian", "Standard")
        trans_val = row.get("transmisi", "-")
        bb_val = row.get("bahan_bakar", "-")

        hasil.append({
            "id_listing": id_list,
            "judul": f"{row['merek']} {row['model']}",
            "varian": varian_val if varian_val != "Standard" else f"{trans_val} • {bb_val}",
            "harga_angka": int(harga_unit),
            "harga_fmt": f"Rp {harga_unit:,.0f}".replace(",", "."),
            "tahun": tahun_str,
            "kilometer": km_str,
            "cc": cc_str,
            "transmisi": trans_val,
            "bahan_bakar": bb_val,
            "status_pajak": row["status_pajak"],
            "kota": row["kota"],
            "hemat": hemat,
            "url_item": url_item,
            "url_cari": url_cari,
        })
    return hasil


def hitung_statistik_grafik_tambahan(df_sumber, jenis):
    # 1. Tren Rata-rata Harga per Tahun
    df_tahun = df_sumber.dropna(subset=["tahun", "harga"]).copy()
    df_tahun["tahun_int"] = df_tahun["tahun"].astype(int)
    df_tahun = df_tahun[(df_tahun["tahun_int"] >= 2012) & (df_tahun["tahun_int"] <= 2025)]
    tren_tahun = (
        df_tahun.groupby("tahun_int")["harga"]
        .mean()
        .sort_index()
        .round(0)
        .to_dict()
    )

    # 2. Sebaran Unit per Kota
    sebaran_kota = (
        df_sumber[df_sumber["kota"] != "Lainnya"]["kota"]
        .value_counts()
        .head(6)
        .to_dict()
    )

    # 3. Rata-rata & Median Harga Berdasarkan Sistem Transmisi
    df_trans = df_sumber[df_sumber["transmisi"].isin(["Automatic", "Manual", "Otomatis", "CVT"])].copy()
    if df_trans.empty:
        df_trans = df_sumber[df_sumber["transmisi"] != "Lainnya"].copy()

    grup_trans = (
        df_trans.groupby("transmisi")["harga"]
        .agg(rata="mean", median="median", jumlah="count")
        .sort_values(by="jumlah", ascending=False)
        .head(4)
    )
    trans_labels = list(grup_trans.index)
    trans_rata = [round(float(v) / 1000000.0, 1) for v in grup_trans["rata"].values]
    trans_median = [round(float(v) / 1000000.0, 1) for v in grup_trans["median"].values]

    # 4. Distribusi Segmen Harga Pasar
    harga_juta = df_sumber["harga"].dropna() / 1000000.0
    if jenis == "mobil":
        bins = [0, 100, 150, 200, 300, 10000]
        segmen_labels = ["< Rp 100 Jt", "Rp 100–150 Jt", "Rp 150–200 Jt", "Rp 200–300 Jt", "> Rp 300 Jt"]
    else:
        bins = [0, 10, 15, 20, 30, 10000]
        segmen_labels = ["< Rp 10 Jt", "Rp 10–15 Jt", "Rp 15–20 Jt", "Rp 20–30 Jt", "> Rp 30 Jt"]

    kategori_harga = pd.cut(harga_juta, bins=bins, labels=segmen_labels, right=False)
    hitung_segmen = kategori_harga.value_counts().reindex(segmen_labels, fill_value=0)
    segmen_values = [int(v) for v in hitung_segmen.values]

    return tren_tahun, sebaran_kota, trans_labels, trans_rata, trans_median, segmen_labels, segmen_values


def hitung_kpi_dataset(df_sumber):
    if df_sumber.empty:
        return {
            "total_unit": 0,
            "median_harga": "Rp 0",
            "total_merek": 0,
            "persen_pajak_hidup": 0,
            "ringkasan_merek": [],
        }

    total_unit = len(df_sumber)
    median_val = float(df_sumber["harga"].median())
    median_harga = f"Rp {median_val:,.0f}".replace(",", ".")
    total_merek = int(df_sumber["merek"].nunique())
    pajak_hidup = int((df_sumber["status_pajak"] == "Hidup").sum())
    persen_pajak_hidup = round((pajak_hidup / total_unit) * 100, 1)

    grup_merek = (
        df_sumber.groupby("merek")
        .agg(
            jumlah_unit=("harga", "count"),
            rata_harga=("harga", "mean"),
            median_harga=("harga", "median"),
        )
        .sort_values(by="jumlah_unit", ascending=False)
        .head(8)
        .reset_index()
    )

    ringkasan_merek = []
    for _, r in grup_merek.iterrows():
        ringkasan_merek.append({
            "merek": r["merek"],
            "jumlah_unit": int(r["jumlah_unit"]),
            "rata_harga": f"Rp {r['rata_harga']:,.0f}".replace(",", "."),
            "median_harga": f"Rp {r['median_harga']:,.0f}".replace(",", "."),
        })

    return {
        "total_unit": total_unit,
        "median_harga": median_harga,
        "total_merek": total_merek,
        "persen_pajak_hidup": persen_pajak_hidup,
        "ringkasan_merek": ringkasan_merek,
    }


def buat_penjelasan_grafik_ai(jenis, grafik_merek, tren_tahun, sebaran_kota, trans_labels, trans_rata, segmen_labels, segmen_values):
    default_insight = {
        "merek": "Grafik menampilkan perbandingan rata-rata harga pasar berdasarkan merek populer di Jawa Timur.",
        "tahun": "Grafik menunjukkan tren kenaikan harga rata-rata seiring semakin mudanya tahun produksi kendaraan.",
        "kota": "Grafik memperlihatkan distribusi jumlah unit kendaraan bekas yang tersedia di kota-kota utama Jawa Timur.",
        "transmisi": "Perbandingan harga rata-rata dan median menunjukkan selisih nilai jual antar jenis sistem transmisi.",
        "segmen": "Distribusi segmen harga memperlihatkan kelompok rentang harga yang paling mendominasi pasar Jawa Timur.",
    }

    if jenis in CACHE_ANALISIS_GRAFIK:
        return CACHE_ANALISIS_GRAFIK[jenis]

    ringkasan_trans = dict(zip(trans_labels, trans_rata))
    ringkasan_segmen = dict(zip(segmen_labels, segmen_values))
    ringkasan_data = (
        f"Kategori: {jenis.upper()} bekas di Jawa Timur.\n"
        f"Data Grafik 1 (Rata-rata harga per merek Rp): {grafik_merek}\n"
        f"Data Grafik 2 (Tren rata-rata harga per tahun Rp): {tren_tahun}\n"
        f"Data Grafik 3 (Jumlah unit per kota): {sebaran_kota}\n"
        f"Data Grafik 4 (Rata-rata harga per sistem transmisi dalam juta Rp): {ringkasan_trans}\n"
        f"Data Grafik 5 (Jumlah unit per segmen harga): {ringkasan_segmen}"
    )
    sys_prompt = (
        "Anda adalah analis pasar otomotif Jawa Timur. Buat penjelasan singkat (1-2 kalimat) untuk masing-masing "
        "dari 5 grafik tersebut dalam bahasa Indonesia yang mudah dipahami tanpa tanda bintang (*) atau markdown. "
        'Kembalikan HANYA JSON valid dengan format persis: '
        '{"merek": "...", "tahun": "...", "kota": "...", "transmisi": "...", "segmen": "..."}'
    )

    jawaban = tanya_groq_ai(sys_prompt, ringkasan_data, max_tokens=450)
    if jawaban and not jawaban.startswith("Gagal"):
        try:
            teks_bersih = jawaban.replace("```json", "").replace("```", "").strip()
            mulai = teks_bersih.find("{")
            akhir = teks_bersih.rfind("}") + 1
            if mulai != -1 and akhir > mulai:
                parsed = json.loads(teks_bersih[mulai:akhir])
                hasil = {
                    "merek": parsed.get("merek", default_insight["merek"]).replace("*", "").strip(),
                    "tahun": parsed.get("tahun", default_insight["tahun"]).replace("*", "").strip(),
                    "kota": parsed.get("kota", default_insight["kota"]).replace("*", "").strip(),
                    "transmisi": parsed.get("transmisi", default_insight["transmisi"]).replace("*", "").strip(),
                    "segmen": parsed.get("segmen", default_insight["segmen"]).replace("*", "").strip(),
                }
                CACHE_ANALISIS_GRAFIK[jenis] = hasil
                return hasil
        except Exception:
            pass

    return default_insight


def halaman_utama(request):
    jenis = request.POST.get("jenis") or request.GET.get("jenis", "mobil")
    if jenis not in ["mobil", "motor"]:
        jenis = "mobil"

    paket_aktif = ARTEFAK[jenis]
    model_aktif = paket_aktif["model"]
    opsi_dropdown = paket_aktif["opsi_dropdown"]
    grafik_merek = paket_aktif["grafik_merek"]
    metrik_model = paket_aktif["metrik"]

    df_sumber = DATASET_REKOMENDASI[jenis]
    (
        tren_tahun,
        sebaran_kota,
        trans_labels,
        trans_rata,
        trans_median,
        segmen_labels,
        segmen_values,
    ) = hitung_statistik_grafik_tambahan(df_sumber, jenis)

    kpi_pasar = hitung_kpi_dataset(df_sumber)
    analisis_grafik_ai = buat_penjelasan_grafik_ai(
        jenis, grafik_merek, tren_tahun, sebaran_kota, trans_labels, trans_rata, segmen_labels, segmen_values
    )

    merek_ke_model = opsi_dropdown.get("merek_ke_model", {})
    daftar_merek = sorted(merek_ke_model.keys()) if merek_ke_model else opsi_dropdown["merek"]

    default_merek = "Toyota" if "Toyota" in daftar_merek else ("Honda" if "Honda" in daftar_merek else daftar_merek[0])
    default_model = (
        merek_ke_model[default_merek][0]
        if default_merek in merek_ke_model and merek_ke_model[default_merek]
        else opsi_dropdown["model"][0]
    )

    input_sebelumnya = {
        "merek": request.POST.get("merek", default_merek).strip(),
        "model": request.POST.get("model", default_model).strip(),
    }

    hasil_prediksi = None
    ringkasan_spek_acuan = None
    saran_deal_ai = None
    daftar_rekomendasi = []
    opsi_transmisi_katalog = []
    pesan_error = None

    try:
        if request.method == "POST":
            merek_in = input_sebelumnya["merek"]
            model_in = input_sebelumnya["model"]

            # Filter berdasarkan Merek & Model agar semua varian transmisi & bahan bakar muncul
            df_cocok = df_sumber[(df_sumber["merek"] == merek_in) & (df_sumber["model"] == model_in)]
            if df_cocok.empty:
                df_cocok = df_sumber[df_sumber["merek"] == merek_in]
            if df_cocok.empty:
                df_cocok = df_sumber

            # Ambil profil acuan pasar dari dataset untuk fitur model ML
            tahun_med = int(df_cocok["tahun"].dropna().median()) if not df_cocok["tahun"].dropna().empty else 2020
            km_med = int(df_cocok["kilometer"].dropna().median()) if not df_cocok["kilometer"].dropna().empty else 40000
            cc_med = int(df_cocok["kapasitas_cc"].dropna().median()) if not df_cocok["kapasitas_cc"].dropna().empty else 150
            trans_mod = df_cocok["transmisi"].mode().iloc[0] if not df_cocok["transmisi"].mode().empty else opsi_dropdown["transmisi"][0]
            bb_mod = df_cocok["bahan_bakar"].mode().iloc[0] if not df_cocok["bahan_bakar"].mode().empty else opsi_dropdown["bahan_bakar"][0]
            pajak_mod = df_cocok["status_pajak"].mode().iloc[0] if not df_cocok["status_pajak"].mode().empty else "Hidup"
            kota_mod = df_cocok["kota"].mode().iloc[0] if not df_cocok["kota"].mode().empty else "Surabaya"

            data_prediksi = pd.DataFrame([{
                "merek": merek_in,
                "model": model_in,
                "transmisi": trans_mod,
                "bahan_bakar": bb_mod,
                "status_pajak": pajak_mod,
                "kota": kota_mod,
                "tahun": tahun_med,
                "kilometer": km_med,
                "kapasitas_cc": cc_med,
            }])[FITUR_KOLOM]

            prediksi_angka = float(model_aktif.predict(data_prediksi)[0])
            hasil_prediksi = f"Rp {prediksi_angka:,.0f}".replace(",", ".")
            ringkasan_spek_acuan = (
                f"Acuan populasi pasar: Tahun {tahun_med} • ±{km_med:,} km • {cc_med} cc • Mayoritas {trans_mod} ({bb_mod})"
            ).replace(",", ".")

            daftar_rekomendasi = format_kartu_rekomendasi(df_cocok, prediksi_angka)
            opsi_transmisi_katalog = sorted({u["transmisi"] for u in daftar_rekomendasi if u["transmisi"] and u["transmisi"] != "Lainnya"})

            ringkasan_unit = "\n".join([
                f"- {u['judul']} ({u['varian']}, {u['transmisi']}, {u['bahan_bakar']}) Tahun {u['tahun']}, Harga {u['harga_fmt']}, {u['kilometer']}, Pajak {u['status_pajak']}, Kota {u['kota']}"
                for u in daftar_rekomendasi[:4]
            ])
            prompt_deal = (
                f"Kendaraan dicari: {merek_in} {model_in} (semua varian transmisi & bahan bakar).\n"
                f"Estimasi Harga Wajar ML: {hasil_prediksi}.\n"
                f"Unit teratas di OLX Jatim:\n{ringkasan_unit}"
            )
            sys_deal = (
                "Anda adalah konsultan pembelian kendaraan bekas di Jawa Timur. Berikan saran singkat 3 poin "
                "(bullet points) bahasa Indonesia: (1) varian/unit mana yang paling layak dipilih dari daftar, "
                "(2) catatan transmisi, pajak, atau kilometer yang perlu diperhatikan, dan (3) 1 tips cek fisik spesifik untuk model ini. Maksimal 110 kata."
            )
            saran_deal_ai = tanya_groq_ai(sys_deal, prompt_deal, max_tokens=280)
        else:
            daftar_rekomendasi = []

    except Exception as e:
        pesan_error = f"Terjadi kesalahan saat memproses rekomendasi: {e}"

    context = {
        "jenis": jenis,
        "is_post": request.method == "POST",
        "opsi": {**opsi_dropdown, "merek": daftar_merek},
        "input_prev": input_sebelumnya,
        "model_terpilih": input_sebelumnya["model"],
        "merek_ke_model_json": json.dumps(merek_ke_model),
        "hasil_prediksi": hasil_prediksi,
        "ringkasan_spek_acuan": ringkasan_spek_acuan,
        "saran_deal_ai": saran_deal_ai.replace("**", "").replace("* ", "• ") if saran_deal_ai else None,
        "insight_grafik": analisis_grafik_ai,
        "rekomendasi": daftar_rekomendasi,
        "opsi_transmisi_katalog": opsi_transmisi_katalog,
        "pesan_error": pesan_error,
        "metrik": {
            "r2": metrik_model["r2"],
            "mae": metrik_model["mae"],
            "mae_fmt": f"Rp {metrik_model['mae']:,.0f}".replace(",", "."),
        },
        "kpi": kpi_pasar,
        "grafik_labels": json.dumps(list(grafik_merek.keys())),
        "grafik_values": json.dumps(list(grafik_merek.values())),
        "tahun_labels": json.dumps([str(k) for k in tren_tahun.keys()]),
        "tahun_values": json.dumps(list(tren_tahun.values())),
        "kota_labels": json.dumps(list(sebaran_kota.keys())),
        "kota_values": json.dumps(list(sebaran_kota.values())),
        "trans_labels_json": json.dumps(trans_labels),
        "trans_rata_json": json.dumps(trans_rata),
        "trans_median_json": json.dumps(trans_median),
        "segmen_labels_json": json.dumps(segmen_labels),
        "segmen_values_json": json.dumps(segmen_values),
    }
    return render(request, "cerdas_app/index.html", context)


@require_GET
def ambil_foto_olx_api(request, id_listing):
    angka_id = id_listing.replace("OLX-", "").strip()
    if angka_id in CACHE_FOTO_OLX:
        return JsonResponse({"image_url": CACHE_FOTO_OLX[angka_id]})

    api_url = f"https://www.olx.co.id/api/items/{angka_id}"
    req = urllib.request.Request(
        api_url,
        headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=4) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            images = data.get("data", {}).get("images", [])
            image_url = None
            if images:
                image_url = (
                    images[0].get("medium", {}).get("url")
                    or images[0].get("big", {}).get("url")
                    or images[0].get("url")
                )
            CACHE_FOTO_OLX[angka_id] = image_url
            return JsonResponse({"image_url": image_url})
    except Exception:
        CACHE_FOTO_OLX[angka_id] = None
        return JsonResponse({"image_url": None})