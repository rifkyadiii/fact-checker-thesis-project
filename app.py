import streamlit as st
import pandas as pd
import shap

from src.model import load_model
from src.predict import predict
from src.utils import compute_shap, waterfall_fig, build_narasi
from src.scraper import scrape_article

# ─── PAGE SETUP ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Detektor Hoaks & Fakta",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── CUSTOM CSS ───────────────────────────────────────────────────────────────
st.markdown("""
    <style>
        .main-title {
            font-size: 3rem;
            font-weight: 800;
            background: -webkit-linear-gradient(45deg, #1e3c72, #2a5298, #ff4b2b);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-align: center;
            margin-bottom: 0px;
            padding-bottom: 0px;
        }
        .sub-title {
            font-size: 1.1rem;
            color: #6c757d;
            text-align: center;
            margin-top: 5px;
            margin-bottom: 2rem;
            font-weight: 500;
        }
        .shap-html-container {
            background-color: #ffffff;
            padding: 1.5rem;
            border-radius: 12px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            color: black !important;
        }
        div[data-testid="stContainer"] {
            border-radius: 10px;
        }
        [data-testid="InputInstructions"] {
            display: none !important;
        }
    </style>
""", unsafe_allow_html=True)

# ─── STATE MANAGEMENT ─────────────────────────────────────────────────────────
if "input_text" not in st.session_state:
    st.session_state.input_text = ""
if "show_scrape_tip" not in st.session_state:
    st.session_state.show_scrape_tip = False

def clear_text():
    st.session_state.input_text = ""
    st.session_state.show_scrape_tip = False

# ─── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Pengaturan")
    st.markdown("Sesuaikan parameter visualisasi model di bawah ini.")
    top_n_display   = st.slider("📊 Jumlah Fitur pada Waterfall Plot", min_value=5,  max_value=20, value=10, step=1)
    table_n_display = st.slider("🗂️ Jumlah Data pada Tabel SHAP",       min_value=5,  max_value=50, value=10, step=1)

    st.markdown("---")
    st.markdown("### ℹ️ Tentang Model")
    st.info("**Base:** IndoBERT (p1)\n\n**Skenario:** 70-20-10\n\n**Explainability:** SHAP")

# ─── LOAD MODEL ───────────────────────────────────────────────────────────────
pipe, _ = load_model()

# ─── MAIN UI ──────────────────────────────────────────────────────────────────
st.markdown('<p class="main-title">Ayok Cek Fakta!</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Sistem Deteksi Hoaks Berbasis IndoBERT dengan Penjelasan SHAP</p>', unsafe_allow_html=True)

st.markdown("### 📰 Masukkan Berita")

# ── Input URL (opsional) ──────────────────────────────────────────────────────
url_col, fetch_col = st.columns([8, 1], vertical_alignment="bottom")

with url_col:
    url_input = st.text_input(
        "URL",
        placeholder="🔗 Tempel link artikel (opsional) — kosongkan jika ingin input teks manual",
        label_visibility="collapsed",
    )
with fetch_col:
    fetch_btn = st.button("🌐 Ambil", width="stretch")

if fetch_btn:
    if not url_input.strip():
        st.warning("⚠️ Masukkan URL terlebih dahulu.")
    else:
        with st.spinner("⏳ Mengambil teks dari halaman web..."):
            scraped_text, scrape_error = scrape_article(url_input)
            if scrape_error:
                st.error(f"❌ {scrape_error}")
            else:
                st.session_state.input_text = scraped_text
                st.session_state.show_scrape_tip = True
                word_count = len(scraped_text.split())
                st.success(f"✅ Berhasil mengambil **{word_count} kata** dari artikel.")
                st.rerun()

# ── Input Teks ────────────────────────────────────────────────────────────────
text_col, action_col = st.columns([8, 1], vertical_alignment="bottom")

with text_col:
    st.text_area(
        "Teks Berita",
        height=180,
        key="input_text",
        label_visibility="collapsed",
        placeholder="Ketik atau tempel teks artikel/berita/pesan berantai WhatsApp yang ingin Anda verifikasi di sini..."
    )

with action_col:
    run = st.button("🚀\nAnalisis", type="primary", width="stretch")
    st.button("🗑️\nHapus", on_click=clear_text, width="stretch")

if st.session_state.show_scrape_tip:
    st.info(
        "📋 **Teks berhasil diambil.** Sebaiknya periksa kembali isinya sebelum dianalisis. "
        "Untuk hasil yang lebih akurat, hapus bagian yang tidak relevan — seperti nama penulis, tanggal, "
        "kategori, tag, atau teks navigasi yang ikut tersalin — dan pertahankan hanya isi narasi beritanya."
    )

# ─── ANALISIS ─────────────────────────────────────────────────────────────────
if run:
    if len(st.session_state.input_text.strip()) < 10:
        st.warning("⚠️ Teks terlalu pendek. Mohon masukkan setidaknya satu kalimat utuh untuk dianalisis.")
    else:
        my_bar = st.progress(0, text="Memulai mesin NLP...")
        st.session_state.show_scrape_tip = False

        my_bar.progress(30, text="Mengekstrak probabilitas dengan IndoBERT...")
        label, prob_hoaks, prob_fakta, num_chunks = predict(st.session_state.input_text, pipe)
        prob_pred = prob_hoaks if label == "Hoaks" else prob_fakta

        my_bar.progress(70, text="Menghitung kontribusi fitur (SHAP Values)...")
        shap_hoaks = compute_shap(st.session_state.input_text, pipe)

        my_bar.progress(100, text="Analisis Selesai!")
        st.toast("Selesai menganalisis teks!", icon="🎉")

        # ── Hasil Prediksi ───────────────────────────────────────────────────
        st.divider()
        if label == "Hoaks":
            st.error("### 🚨 TEKS TERINDIKASI HOAKS!")
        else:
            st.success("### ✅ TEKS CENDERUNG FAKTA")

        met_col1, met_col2, met_col3 = st.columns(3)
        with met_col1:
            st.metric("Confidence Score", f"{prob_pred*100:.2f}%",
                      delta="Sangat Yakin" if prob_pred > 0.8 else "Kurang Yakin", delta_color="normal")
        with met_col2:
            st.metric("Probabilitas Hoaks", f"{prob_hoaks*100:.2f}%")
        with met_col3:
            st.metric("Probabilitas Fakta", f"{prob_fakta*100:.2f}%")

        # ── Narasi NLG ───────────────────────────────────────────────────────
        st.markdown("### 💡 Insight")
        st.info(build_narasi(label, prob_pred, shap_hoaks))

        # ── Visualisasi SHAP ─────────────────────────────────────────────────
        st.markdown("### 🎨 Visualisasi SHAP")
        tab1, tab2, tab3 = st.tabs(["📝 Text Highlight", "📉 Waterfall Plot", "📊 Data Tabel SHAP"])

        with tab1:
            st.markdown(
                "Bagian berwarna <span style='color:red; font-weight:bold;'>Merah</span> menarik keputusan ke arah **Hoaks**, "
                "sedangkan <span style='color:blue; font-weight:bold;'>Biru</span> menarik ke arah **Fakta**.",
                unsafe_allow_html=True
            )
            try:
                html_out = shap.plots.text(shap_hoaks[0], display=False)
                if html_out:
                    st.iframe(f'<div class="shap-html-container">{html_out}</div>', height=350)
                else:
                    st.info("Text plot tidak tersedia untuk teks ini.")
            except Exception as e:
                st.warning(f"Text plot gagal di-render: {e}")

        with tab2:
            st.markdown("Grafik ini menunjukkan secara bertahap bagaimana setiap fitur menambah atau mengurangi probabilitas prediksi.")
            buf = waterfall_fig(shap_hoaks, top_n_display, label, prob_pred)
            col_left, col_center, col_right = st.columns([1, 3, 1])
            with col_center:
                st.image(buf, width="stretch")

        with tab3:
            st.markdown("Nilai kontribusi setiap token terhadap prediksi model. Nilai positif = kecenderungan Hoaks, negatif = kecenderungan Fakta.")
            vals   = shap_hoaks[0].values
            toks   = shap_hoaks[0].data
            ranked = sorted(zip(toks, vals), key=lambda x: abs(x[1]), reverse=True)[:table_n_display]
            df_shap = pd.DataFrame(ranked, columns=["Token", "SHAP Value"])

            def color_shap(val):
                if val > 0: return "background-color: #ffcccc"
                if val < 0: return "background-color: #ccccff"
                return "background-color: transparent"

            styled_df = df_shap.style.map(color_shap, subset=["SHAP Value"]).format({"SHAP Value": "{:.10f}"})
            st.dataframe(styled_df, width="stretch", height=300)