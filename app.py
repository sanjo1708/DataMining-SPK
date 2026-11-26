import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import joblib
import numpy as np
import streamlit as st

# Konfigurasi Halaman (Mobile-friendly layout)
st.set_page_config(page_title="EduFit", page_icon="🎓", layout="centered")

@st.cache_resource
def load_artifacts():
    return joblib.load("model_studisehat.joblib")

artifacts = load_artifacts()

def hitung_spk_dan_klaster(c1, c2, c3, c4):
    kriteria = np.array([c1, c2, c3, c4], dtype=float)
    
    # 1. Normalisasi SAW (semua kriteria bersifat Benefit)
    max_vals = np.array(artifacts["saw_max_values"], dtype=float)
    weights = np.array(artifacts["saw_weights"], dtype=float)
    
    r_norm = np.clip(kriteria / max_vals, 0, 1)
    skor_saw = float(np.sum(r_norm * weights))
    
    # 2. Prediksi Klaster K-Means menggunakan fitur C1-C3
    fitur_kmeans = np.array([[c1, c2, c3]], dtype=float)
    if "scaler" in artifacts and artifacts["scaler"] is not None:
        fitur_kmeans = artifacts["scaler"].transform(fitur_kmeans)
        
    cluster_id = int(artifacts["kmeans_model"].predict(fitur_kmeans)[0])
    label_klaster = artifacts["cluster_labels"][cluster_id]
    
    return skor_saw, r_norm, cluster_id, label_klaster

# Inisialisasi Session State untuk navigasi antar tab (opsional untuk logika lanjutan)
if 'hitung' not in st.session_state:
    st.session_state.hitung = False

# Membuat 3 Navigasi (Tab) meniru Bottom Navigation Mobile App
tab1, tab2, tab3 = st.tabs(["🏠 Beranda", "📝 Kuesioner", "📊 Hasil SPK"])

# ==========================================
# LAYAR 1: BERANDA / DASHBOARD
# ==========================================
with tab1:
    st.title("Halo, Mahasiswa! 👋")
    st.caption('"Keseimbangan fisik dan mental adalah kunci keberhasilan akademik."')

    st.subheader("Skor SPK Terkini")
    col1, col2 = st.columns(2)

    if st.session_state.hitung:
        skor_tampil = f"{st.session_state.skor_saw:.2f}"
        status_tampil = st.session_state.label_klaster
    else:
        skor_tampil = "-"
        status_tampil = "Belum diisi"

    col1.metric(label="Skor Preferensi SAW", value=skor_tampil)
    col2.metric(label="Status Profil", value=status_tampil)

    st.divider()

    st.write("📊 **Distribusi Bobot Kriteria (Metode SAW)**")
    w = artifacts["saw_weights"]

    st.write(f"C1 - Jasmani (Benefit): {w[0]*100:.1f}%")
    st.progress(float(w[0]))

    st.write(f"C2 - Mental (Benefit): {w[1]*100:.1f}%")
    st.progress(float(w[1]))

    st.write(f"C3 - Kebiasaan Akademik (Benefit): {w[2]*100:.1f}%")
    st.progress(float(w[2]))

    st.write(f"C4 - Performa Akademik (Benefit): {w[3]*100:.1f}%")
    st.progress(float(w[3]))

# ==========================================
# LAYAR 2: KUESIONER PENILAIAN
# ==========================================
with tab2:
    st.subheader("📝 Kuesioner Gaya Hidup & Akademik")
    st.write("Pilih salah satu jawaban di tiap pertanyaan sesuai kondisi kamu sekarang.")

    map_tidur = {
        "< 5 jam": 1,
        "5 - 6 jam": 2,
        "6 - 7 jam": 3,
        "7 - 8 jam": 4,
        "> 8 jam": 5
    }
    map_olahraga = {
        "Tidak pernah": 1,
        "1 kali seminggu": 2,
        "2 kali seminggu": 3,
        "3 - 4 kali seminggu": 4,
        "≥ 5 kali seminggu": 5
    }
    map_stres = {
        "Sangat Tinggi": 1,
        "Tinggi": 2,
        "Sedang": 3,
        "Rendah": 4,
        "Sangat Rendah": 5
    }
    map_belajar = {
        "< 1 jam/hari": 1,
        "1 - 2 jam/hari": 2,
        "2 - 3 jam/hari": 3,
        "3 - 4 jam/hari": 4,
        "> 4 jam/hari": 5
    }
    map_disiplin = {
        "Sering terlambat & menunda tugas": 1,
        "Kadang terlambat mengumpulkan tugas": 2,
        "Cukup tepat waktu": 3,
        "Selalu tepat waktu": 4,
        "Sangat terencana & selesai lebih awal": 5
    }
    map_ipk = {
        "< 2.50": 1.0,
        "2.50 - 2.99": 2.0,
        "Belum memiliki IPK": 3.0,
        "3.00 - 3.49": 4.0,
        "3.50 - 4.00": 5.0
    }

    with st.form("form_kuesioner"):
        with st.container(border=True):
            st.markdown("#### 🏃 C1 — Kesehatan Jasmani")
            q_tidur = st.radio("1. Rata-rata durasi tidur per hari:", list(map_tidur.keys()), index=None)
            q_olahraga = st.radio("2. Frekuensi olahraga dalam seminggu:", list(map_olahraga.keys()), index=None)

        with st.container(border=True):
            st.markdown("#### 🧠 C2 — Kesehatan Mental")
            q_stres = st.radio("3. Tingkat stres akademik & keseharian:", list(map_stres.keys()), index=None)

        with st.container(border=True):
            st.markdown("#### 📚 C3 — Kebiasaan Akademik")
            q_belajar = st.radio("4. Durasi belajar mandiri di luar jam kuliah:", list(map_belajar.keys()), index=None)
            q_disiplin = st.radio("5. Kedisiplinan manajemen waktu & tugas:", list(map_disiplin.keys()), index=None)

        with st.container(border=True):
            st.markdown("#### 🎓 C4 — Performa Akademik")
            q_ipk = st.radio("6. Kategori IPK saat ini:", list(map_ipk.keys()), index=None)

        submitted = st.form_submit_button("Hitung Skor & Analisis Profil", use_container_width=True)

    if submitted:
        if None in (q_tidur, q_olahraga, q_stres, q_belajar, q_disiplin, q_ipk):
            st.warning("⚠️ Masih ada pertanyaan yang belum dipilih. Lengkapi semua pilihan dulu ya!")
        else:
            c1 = (map_tidur[q_tidur] + map_olahraga[q_olahraga]) / 2.0
            c2 = float(map_stres[q_stres])
            c3 = (map_belajar[q_belajar] + map_disiplin[q_disiplin]) / 2.0
            c4 = float(map_ipk[q_ipk])

            skor_saw, r_norm, cluster_id, label_klaster = hitung_spk_dan_klaster(c1, c2, c3, c4)

            st.session_state.hitung = True
            st.session_state.skor_saw = skor_saw
            st.session_state.r_norm = r_norm
            st.session_state.cluster_id = cluster_id
            st.session_state.label_klaster = label_klaster
            st.session_state.nilai_kriteria = [c1, c2, c3, c4]

            st.toast("✅ Data berhasil diproses! Cek tab Beranda atau Hasil SPK.")
            st.rerun()
                                
# ==========================================
# LAYAR 3: HASIL SPK & REKOMENDASI
# ==========================================
with tab3:
    st.subheader("📊 Hasil Analisis SPK & Profil Klaster")

    if not st.session_state.hitung:
        st.info("Belum ada data yang dihitung. Silakan isi form di tab **Kuesioner** dulu ya!")
    else:
        skor = st.session_state.skor_saw
        r_norm = st.session_state.r_norm
        cluster_id = st.session_state.cluster_id
        label_klaster = st.session_state.label_klaster
        nilai_asli = st.session_state.nilai_kriteria

        # Tentukan kategori status berdasarkan rentang skor SAW
        if skor >= 0.80:
            kategori_saw = "Sangat Baik (Pertahankan)"
        elif skor >= 0.60:
            kategori_saw = "Cukup Baik (Perlu Optimasi)"
        else:
            kategori_saw = "Perlu Perhatian Khusus"

        # Kartu Ringkasan Utama
        col1, col2 = st.columns(2)
        col1.metric("Skor Akhir SAW", f"{skor:.2f}", kategori_saw)
        col2.metric("Klaster K-Means", f"Klaster {cluster_id}", label_klaster)

        st.divider()

        # Grafik Batang Nilai Normalisasi Kriteria (Plotly)
        st.write("**Profil Normalisasi Kriteria (C1 - C4)**")
        kriteria_labels = [
            "C1: Jasmani",
            "C2: Mental",
            "C3: Kebiasaan Akademik",
            "C4: Performa Akademik"
        ]

        fig = go.Figure(
            data=[
                go.Bar(
                    x=kriteria_labels,
                    y=r_norm,
                    text=[f"{val:.2f}" for val in r_norm],
                    textposition="auto",
                    marker_color=["#2E86AB", "#A23B72", "#F18F01", "#06A77D"]
                )
            ]
        )
        fig.update_layout(
            yaxis=dict(title="Nilai Normalisasi (0 - 1)", range=[0, 1.1]),
            xaxis=dict(title=""),
            height=340,
            margin=dict(l=20, r=20, t=20, b=20)
        )
        st.plotly_chart(fig, use_container_width=True)

        st.divider()

        # Rekomendasi Dinamis Berdasarkan Klaster & Kriteria Terendah
        st.write("**💡 Rekomendasi Tindakan Personal**")

        rekomendasi_klaster = {
            0: "Tingkatkan konsistensi jam belajar mandiri di luar kelas dan bikin jadwal prioritas tugas mingguan biar nggak keteteran.",
            1: "Pertahankan ritme istirahat dan jadwal belajarmu yang udah seimbang. Kamu bisa mulai nambah target akademik atau kegiatan pengembangan diri.",
            2: "Fokus perbaiki kualitas tidur dan manajemen stres harian. Kurangi begadang deket tenggat tugas biar performa kuliah tetap stabil."
        }

        saran_utama = rekomendasi_klaster.get(
            cluster_id,
            "Jaga keseimbangan antara waktu istirahat, aktivitas fisik, dan jadwal belajar harian."
        )
        st.success(f"**Fokus Utama ({label_klaster}):** {saran_utama}")

        # Deteksi kriteria dengan skor paling rendah buat saran spesifik
        idx_terendah = int(np.argmin(r_norm))
        saran_spesifik = [
            "Aspek **Jasmani (C1)** paling rendah — coba tambah durasi tidur berkualitas dan olahraga minimal 2-3 kali seminggu.",
            "Aspek **Mental (C2)** paling rendah — luangkan waktu istirahat sejenak dari layar dan kelola beban stres sebelum menumpuk.",
            "Aspek **Kebiasaan Akademik (C3)** paling rendah — biasakan nyicil materi kuliah lebih awal daripada sistem kebut semalam.",
            "Aspek **Performa Akademik (C4)** paling rendah — evaluasi metode belajarmu dan manfaatkan sesi diskusi atau konsultasi dosen."
        ]
        st.warning(f"**Prioritas Perbaikan:** {saran_spesifik[idx_terendah]}")

        # Tombol Unduh Laporan Hasil Analisis
        ringkasan_teks = (
            f"LAPORAN HASIL ANALISIS STUDISEHAT SPK\n"
            f"=====================================\n"
            f"Skor Preferensi SAW : {skor:.4f} ({kategori_saw})\n"
            f"Profil Klaster      : Klaster {cluster_id} - {label_klaster}\n\n"
            f"Rincian Skor Kriteria (Skala 1-5):\n"
            f"- C1 Kesehatan Jasmani    : {nilai_asli[0]:.2f} (Normalisasi: {r_norm[0]:.2f})\n"
            f"- C2 Kesehatan Mental     : {nilai_asli[1]:.2f} (Normalisasi: {r_norm[1]:.2f})\n"
            f"- C3 Kebiasaan Akademik   : {nilai_asli[2]:.2f} (Normalisasi: {r_norm[2]:.2f})\n"
            f"- C4 Performa Akademik    : {nilai_asli[3]:.2f} (Normalisasi: {r_norm[3]:.2f})\n\n"
            f"Rekomendasi Utama:\n{saran_utama}\n"
        )

        st.download_button(
            label="📥 Unduh Ringkasan Hasil (.txt)",
            data=ringkasan_teks,
            file_name="hasil_analisis_studisehat.txt",
            mime="text/plain",
            use_container_width=True
        )