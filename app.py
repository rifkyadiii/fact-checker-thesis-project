import streamlit as st
import torch
import shap
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import io
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification, pipeline

# ─── CONFIG ───────────────────────────────────────────────────────────────────
MODEL_PATH  = "rifkyadiii/best_models_70_20_10"   
BASE_MODEL  = "indobenchmark/indobert-base-p1"
MAX_LEN     = 128
TOP_N       = 3                        

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
        /* Judul utama dengan efek gradient */
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
        /* Memaksa text plot SHAP agar backgroundnya putih di dark mode */
        .shap-html-container { 
            background-color: #ffffff; 
            padding: 1.5rem; 
            border-radius: 12px; 
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            color: black !important;
        }
        /* Mengubah tampilan border container */
        div[data-testid="stContainer"] {
            border-radius: 10px;
        }
    </style>
""", unsafe_allow_html=True)

# ─── STATE MANAGEMENT ─────────────────────────────────────────────────────────
if "input_text" not in st.session_state:
    st.session_state.input_text = ""

def clear_text():
    st.session_state.input_text = ""

# ─── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Pengaturan")
    st.markdown("Sesuaikan parameter visualisasi model di bawah ini.")
    top_n_display = st.slider("📊 Jumlah Fitur pada Waterfall Plot", min_value=5, max_value=20, value=10, step=1)
    table_n_display = st.slider("🗂️ Jumlah Data pada Tabel SHAP", min_value=5, max_value=50, value=10, step=1)
    
    st.markdown("---")
    st.markdown("### ℹ️ Tentang Model")
    st.info("**Base:** IndoBERT (p1)\n\n**Skenario:** 70-20-10\n\n**Explainability:** SHAP")

# ─── LOAD MODEL (cached) ──────────────────────────────────────────────────────
@st.cache_resource(show_spinner="🔄 Memuat model AI ke memori (Mohon tunggu)...")
def load_model():
    tokenizer = BertTokenizer.from_pretrained(BASE_MODEL)
    model     = BertForSequenceClassification.from_pretrained(MODEL_PATH)
    device    = 0 if torch.cuda.is_available() else -1
    pipe      = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        device=device,
        top_k=None,
    )
    explainer = shap.Explainer(pipe)
    return pipe, explainer

pipe, explainer = load_model()

# ─── HELPER: PREDICT & SHAP ───────────────────────────────────────────────────
def predict(text: str):
    out = pipe(text, truncation=True, max_length=MAX_LEN)[0]
    scores = {item["label"]: item["score"] for item in out}
    prob_hoaks = scores.get("LABEL_1", 0.0)
    prob_fakta = scores.get("LABEL_0", 0.0)
    label = "Hoaks" if prob_hoaks >= prob_fakta else "Fakta"
    return label, prob_hoaks, prob_fakta

def compute_shap(text: str):
    sv = explainer([text])
    return sv[:, :, 1]   

def waterfall_fig(shap_hoaks, max_display: int, label: str, prob: float):
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    
    shap.plots.waterfall(shap_hoaks[0], max_display=max_display, show=False)
    fig = plt.gcf()
    fig.suptitle(
        f"Waterfall Plot — {label}",
        fontsize=14, fontweight="bold", y=1.05, color="#1e3c72"
    )
    
    for text in fig.texts: text.set_color('black')
    ax.tick_params(colors='black')
    
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=300, bbox_inches="tight", facecolor='white')
    plt.close(fig)
    buf.seek(0)
    return buf


# ─── HELPER: NARASI NLG ──────────────────────────────────────────────────────
def build_narasi(label: str, prob: float, shap_hoaks) -> str:
    """Rule-based NLG untuk penjelasan naratif klasifikasi."""

    vals = shap_hoaks[0].values
    toks = shap_hoaks[0].data

    ranked = sorted(zip(toks, vals), key=lambda x: abs(x[1]), reverse=True)
    top = ranked[:TOP_N]

    kelas_pred = label
    kelas_lawan = "Fakta" if label == "Hoaks" else "Hoaks"

    def arah(val, label_pred):
        if label_pred == "Hoaks":
            return val > 0
        else:
            return val < 0

    conf_pct = prob * 100

    if conf_pct >= 90:
        tingkat_keyakinan = "sangat tinggi"
    elif conf_pct >= 75:
        tingkat_keyakinan = "tinggi"
    elif conf_pct >= 60:
        tingkat_keyakinan = "cukup"
    else:
        tingkat_keyakinan = "relatif rendah"

    kalimat_pembuka = (
        f"Berdasarkan tinjauan model, berita ini diprediksi sebagai **{kelas_pred}** "
        f"dengan confidence score **{conf_pct:.2f}%** — tingkat keyakinan model tergolong *{tingkat_keyakinan}*."
    )

    # Ambil top 2 token
    tok1, val1 = top[0]
    tok2, val2 = top[1]

    tok1, tok2 = tok1.strip(), tok2.strip()

    searah1 = arah(val1, kelas_pred)
    searah2 = arah(val2, kelas_pred)

    if searah1 and searah2:
        kalimat_fitur = (
            f"Keputusan tersebut terutama dipengaruhi oleh fitur **'{tok1}'** dan **'{tok2}'**, "
            f"keduanya dinilai memberikan kontribusi besar yang mengarahkan model ke kategori **{kelas_pred}**."
        )
    elif searah1 and not searah2:
        kalimat_fitur = (
            f"fitur **'{tok1}'** menjadi pendorong utama ke arah **{kelas_pred}**, "
            f"sementara fitur **'{tok2}'** justru memberikan sinyal berlawanan yang mengarah ke **{kelas_lawan}**."
        )
    elif not searah1 and searah2:
        kalimat_fitur = (
            f"fitur **'{tok2}'** memperkuat prediksi ke arah **{kelas_pred}**, "
            f"sedangkan fitur **'{tok1}'** memberikan sinyal yang tidak sejalan, mengarah ke **{kelas_lawan}**."
        )
    else:
        kalimat_fitur = (
            f"Menariknya, fitur **'{tok1}'** dan **'{tok2}'** keduanya memberikan sinyal berlawanan dari prediksi utama, "
            f"namun pengaruhnya belum cukup untuk mengubah keputusan model."
        )

    # Token ke-3 (opsional)
    kalimat_kondisional = ""
    if len(top) >= 3:
        tok3, val3 = top[2]
        tok3 = tok3.strip()

        searah3 = arah(val3, kelas_pred)

        if searah3:
            kalimat_kondisional = (
                f"Selain itu, keberadaan fitur **'{tok3}'** turut memperkuat kecenderungan model "
                f"dalam mengklasifikasikan teks ini ke dalam kelas **{kelas_pred}**."
            )
        else:
            kalimat_kondisional = (
                f"Di sisi lain, fitur **'{tok3}'** memberikan sinyal yang tidak sejalan dengan prediksi utama — "
                f"namun pengaruhnya relatif kecil sehingga tidak mengubah hasil keputusan model."
            )

    # Penutup
    if kelas_pred == "Hoaks":
        if conf_pct >= 75:
            kalimat_penutup = (
                "⚠️ **Rekomendasi:** Secara keseluruhan, pola linguistik dalam teks ini menunjukkan karakteristik yang umumnya ditemukan pada konten hoaks. "
                "Disarankan untuk memverifikasi informasi ini melalui sumber terpercaya sebelum disebarluaskan."
            )
        else:
            kalimat_penutup = (
                "⚠️ **Rekomendasi:** Model mendeteksi kemungkinan hoaks, namun dengan tingkat keyakinan yang belum terlalu tinggi. "
                "Tetap lakukan verifikasi mandiri terhadap klaim-klaim dalam teks ini."
            )
    else:
        if conf_pct >= 75:
            kalimat_penutup = (
                "⚠️ **Rekomendasi:** Secara keseluruhan, pola linguistik dalam teks ini konsisten dengan berita faktual. "
                "Meski demikian, verifikasi tetap dianjurkan untuk memastikan akurasi informasi."
            )
        else:
            kalimat_penutup = (
                "⚠️ **Rekomendasi:** Model cenderung mengklasifikasikan teks ini sebagai fakta, namun dengan keyakinan yang moderat — "
                "tetap bijak dalam menyimpulkan."
            )

    bagian = [kalimat_pembuka, kalimat_fitur]

    if kalimat_kondisional:
        bagian.append(kalimat_kondisional)

    bagian.append(kalimat_penutup)

    return "\n\n".join(bagian)

# ─── MAIN UI ──────────────────────────────────────────────────────────────────
st.markdown('<p class="main-title">Ayok Cek Fakta!</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Sistem Deteksi Hoaks Berbasis IndoBERT dengan Penjelasan SHAP</p>', unsafe_allow_html=True)

st.markdown("### 📰 Masukkan Berita")
text_col, action_col = st.columns([8, 1])

with text_col:
    st.text_area(
        "Teks Berita",
        height=180,
        key="input_text",
        label_visibility="collapsed",
        placeholder="Ketik atau tempel teks artikel/berita/pesan berantai WhatsApp yang ingin Anda verifikasi di sini..."
    )

with action_col:
    st.markdown("<br>", unsafe_allow_html=True) # Spacer
    run = st.button("🚀\nAnalisis", type="primary", use_container_width=True)
    st.button("🗑️\nHapus", on_click=clear_text, use_container_width=True)

if run:
    if len(st.session_state.input_text.strip()) < 10:
        st.warning("⚠️ Teks terlalu pendek. Mohon masukkan setidaknya satu kalimat utuh untuk dianalisis.")
    else:
        # Progress Bar untuk UX yang lebih baik
        progress_text = "Memulai mesin NLP..."
        my_bar = st.progress(0, text=progress_text)
        
        my_bar.progress(30, text="Mengekstrak probabilitas dengan IndoBERT...")
        label, prob_hoaks, prob_fakta = predict(st.session_state.input_text)
        prob_pred = prob_hoaks if label == "Hoaks" else prob_fakta

        my_bar.progress(70, text="Menghitung kontribusi fitur (SHAP Values)...")
        shap_hoaks = compute_shap(st.session_state.input_text)
        
        my_bar.progress(100, text="Analisis Selesai!")
        st.toast('Selesai menganalisis teks!', icon='🎉')
        

        # ── Hasil Prediksi (DASHBOARD METRICS) ─────────────────────────────
        st.divider()
        
        if label == "Hoaks":
            st.error("### 🚨 TEKS TERINDIKASI HOAKS!")
        else:
            st.success("### ✅ TEKS CENDERUNG FAKTA")

        # Tampilan Metric 3 Kolom
        met_col1, met_col2, met_col3 = st.columns(3)
        with met_col1:
            st.metric(label="Confidence Score", value=f"{prob_pred*100:.2f}%", 
                      delta="Sangat Yakin" if prob_pred > 0.8 else "Kurang Yakin", delta_color="normal")
        with met_col2:
            st.metric(label="Probabilitas Hoaks", value=f"{prob_hoaks*100:.2f}%")
        with met_col3:
            st.metric(label="Probabilitas Fakta", value=f"{prob_fakta*100:.2f}%")

        # ── Penjelasan Naratif (INSIGHT) ────────────────────────────────
        st.markdown("### 💡 Insight")
        narasi = build_narasi(label, prob_pred, shap_hoaks)
        st.info(narasi)

        # ── Visualisasi SHAP ───────────────────────────────────────────────
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
                    html_wrapper = f'<div class="shap-html-container">{html_out}</div>'
                    st.components.v1.html(html_wrapper, height=350, scrolling=True)
                else:
                    st.info("Text plot tidak tersedia untuk teks ini.")
            except Exception as e:
                st.warning(f"Text plot gagal di-render: {e}")

        with tab2:
            st.markdown("Grafik ini menunjukkan secara bertahap bagaimana setiap fitur menambah atau mengurangi probabilitas prediksi.")
            buf = waterfall_fig(shap_hoaks, top_n_display, label, prob_pred)
            
            col_left, col_center, col_right = st.columns([1, 3, 1])
            
            with col_center:
                st.image(buf, use_container_width=True)
                
        with tab3:
            st.markdown("Nilai kontribusi setiap token terhadap prediksi model. Nilai positif menunjukkan kecenderungan ke arah hoaks, sedangkan nilai negatif menunjukkan kecenderungan ke arah fakta.")
            vals  = shap_hoaks[0].values
            toks  = shap_hoaks[0].data
            # Ambil semua, tidak cuma 15 agar lebih lengkap
            ranked = sorted(zip(toks, vals), key=lambda x: abs(x[1]), reverse=True)[:table_n_display]
            df_shap = pd.DataFrame(ranked, columns=["Token", "SHAP Value"])
            
            # Styling Dataframe (Merah untuk > 0, Biru untuk < 0)
            def color_shap(val):
                color = '#ffcccc' if val > 0 else '#ccccff'
                if val == 0: color = 'transparent'
                return f'background-color: {color}'

            styled_df = df_shap.style.map(color_shap, subset=["SHAP Value"]).format({"SHAP Value": "{:.10f}"})
            st.dataframe(styled_df, use_container_width=True, height=300)