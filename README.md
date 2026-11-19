# 🚗 HAFITOMOTIVE — Drive with Better Insights.

**HAFITOMOTIVE** adalah aplikasi web cerdas berbasis Machine Learning dan Generative AI untuk mengestimasi harga pasar wajar kendaraan bekas (mobil dan motor) serta memberikan rekomendasi unit perbandingan berdasarkan data iklan riil wilayah Jawa Timur.

🔗 **Live Website:** [https://sistem-cerdas-hafitomotive.vercel.app](https://sistem-cerdas-hafitomotive.vercel.app)  
📦 **Repository:** [https://github.com/k3vin16/SistemCerdas-Hafitomotive](https://github.com/k3vin16/SistemCerdas-Hafitomotive)

---

## 👤 Informasi Akademik

* **Nama Mahasiswa:** Hafidh Gilang Ramadhan
* **NIM:** 253307047
* **Program Studi / Kelas:** Teknologi Informasi / 3B
* **Mata Kuliah:** Sistem Cerdas
* **Dosen Pengampu:** Muhammad Syaeful Fajar, S.Pd., M.Kom.
* **Tema Proyek:** Model regresi/klasifikasi dari dataset terbuka pilihan + form input + hasil prediksi + grafik data (Aplikasi Prediksi ML)

---

## ✨ Fitur Utama

1. **Prediksi Harga Pasar Wajar (Random Forest Regressor)**
   * Mendukung dua kategori kendaraan secara terpisah: **Mobil Bekas** dan **Motor Bekas**.
   * Cukup memilih **Merek** dan **Model / Seri** kendaraan (dengan *cascading dropdown* dinamis) untuk mendapatkan estimasi harga pasar wajar beserta profil spesifikasi mayoritas di pasar.
2. **AI Automotive Advisor (Groq LLM API)**
   * Memberikan ringkasan analisis pasar, poin pengecekan fisik, dan rekomendasi pembelian secara kontekstual sesuai merek dan model kendaraan yang dipilih.
3. **Katalog Perbandingan Unit Riil OLX Jawa Timur**
   * Menampilkan daftar unit tersedia yang relevan dengan hasil prediksi, lengkap dengan status perbandingan harga (**Di Bawah Estimasi** / **Di Atas Estimasi**), filter transmisi, status pajak, dan pencarian cepat berdasarkan kota atau varian.
4. **Market Data Intelligence Dashboard**
   * Dilengkapi 5 visualisasi interaktif menggunakan **Chart.js** (rata-rata harga per merek, tren harga per tahun produksi, sebaran stok per kota di Jawa Timur, perbandingan harga per sistem transmisi, dan distribusi segmen harga) serta tabel ringkasan statistik merek populer.
5. **Desain Responsif & Modern**
   * Antarmuka responsif berbasis **Bootstrap 5** yang optimal diakses melalui perangkat desktop maupun smartphone.

---

## 🛠️ Teknologi & Arsitektur

* **Backend Framework:** Python 3.12, Django 5
* **Machine Learning & Data Processing:** Scikit-Learn (`RandomForestRegressor`), Pandas, NumPy, Joblib
* **AI Reasoning Engine:** Groq API
* **Frontend & Visualisasi:** HTML5, CSS3, Bootstrap 5, Chart.js, Lucide Icons
* **Deployment & Static Serving:** Vercel Serverless (`sin1` Singapore Region), WhiteNoise

---

## 📂 Struktur Direktori Proyek

```text
proyek_sistem_cerdas/
├── cerdas_app/                     # Aplikasi utama Django (views, urls, templates, static)
│   ├── static/cerdas_app/assets/   # Aset logo HAFITOMOTIVE
│   └── templates/cerdas_app/       # Template antarmuka web (index.html)
├── proyek_sistem_cerdas/           # Konfigurasi proyek Django (settings, urls, wsgi)
├── dataset_olx_mobil_real.csv      # Dataset riil mobil bekas Jawa Timur
├── dataset_olx_motor_real.csv      # Dataset riil motor bekas Jawa Timur
├── latih_model.py                  # Skrip pelatihan & evaluasi model Random Forest
├── model.joblib                    # Model ML terlatih untuk inferensi cepat
├── requirements.txt                # Daftar dependensi Python
├── vercel.json                     # Konfigurasi deployment Vercel Serverless
└── manage.py                       # Utilitas manajemen perintah Django
```

---

## 🔒 Keamanan Environment Variable

Proyek ini menerapkan praktik keamanan standar untuk melindungi kredensial API:
* Kunci API Groq (`GROQ_API_KEY`) disimpan menggunakan **Environment Variables** baik di lingkungan lokal (`.env`) maupun di pengaturan *Environment Variables* pada dashboard Vercel.
* File `.env`, database lokal (`db.sqlite3`), serta direktori virtual environment (`.venv/`) dikecualikan dari pelacakan Git melalui file `.gitignore` agar tidak terekspos di repositori publik.

---

## 💻 Cara Menjalankan di Komputer Lokal

1. **Clone repositori ini:**
   ```bash
   git clone https://github.com/k3vin16/SistemCerdas-Hafitomotive.git
   cd SistemCerdas-Hafitomotive
   ```

2. **Buat dan aktifkan virtual environment:**
   ```bash
   python -m venv .venv
   # Windows PowerShell:
   .\.venv\Scripts\Activate.ps1
   # Linux/macOS:
   source .venv/bin/activate
   ```

3. **Instal seluruh dependensi:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Konfigurasi file `.env`:**
   Buat file `.env` di direktori utama proyek dan tambahkan API key Groq Anda:
   ```env
   GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxxxxxxxxxx
   ```

5. **Jalankan server lokal Django:**
   ```bash
   python manage.py collectstatic --noinput
   python manage.py runserver
   ```
   Buka browser dan akses `http://127.0.0.1:8000/`.