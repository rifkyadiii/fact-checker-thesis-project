import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import shap

from src.config import TOP_N
from src.predict import chunk_text


# ─── SHAP COMPUTATION ─────────────────────────────────────────────────────────

def compute_shap(text: str, pipe) -> list:
    """Menghitung SHAP values, menggabungkan semua chunk menjadi satu Explanation."""
    chunks = chunk_text(text)
    all_values, all_data, all_base = [], [], []

    for chunk in chunks:
        fresh_explainer = shap.Explainer(pipe)
        sv  = fresh_explainer([chunk])
        exp = sv[:, :, 1][0]           # indeks 1 = kelas Hoaks
        all_values.append(exp.values)
        all_data.append(exp.data)
        all_base.append(float(exp.base_values))

    combined = shap.Explanation(
        values      = np.concatenate(all_values),
        base_values = float(np.mean(all_base)),
        data        = np.concatenate(all_data),
    )
    return [combined]   # dibungkus list agar shap_hoaks[0] tetap berfungsi


# ─── WATERFALL PLOT ────────────────────────────────────────────────────────────

def waterfall_fig(shap_hoaks, max_display: int, label: str, prob: float) -> io.BytesIO:
    """Menghasilkan waterfall plot sebagai BytesIO (PNG)."""
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    shap.plots.waterfall(shap_hoaks[0], max_display=max_display, show=False)
    fig = plt.gcf()
    fig.suptitle(
        f"Waterfall Plot — {label}",
        fontsize=14, fontweight="bold", y=1.05, color="#1e3c72"
    )

    for text in fig.texts:
        text.set_color("black")
    ax.tick_params(colors="black")

    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return buf


# ─── NARASI NLG ───────────────────────────────────────────────────────────────

def build_narasi(label: str, prob: float, shap_hoaks) -> str:
    """Membangun teks penjelasan naratif berbasis SHAP values."""
    vals   = shap_hoaks[0].values
    toks   = shap_hoaks[0].data
    ranked = sorted(zip(toks, vals), key=lambda x: abs(x[1]), reverse=True)
    top    = ranked[:TOP_N]

    kelas_pred  = label
    kelas_lawan = "Fakta" if label == "Hoaks" else "Hoaks"
    conf_pct    = prob * 100

    def arah(val, label_pred):
        return val > 0 if label_pred == "Hoaks" else val < 0

    # ── Tingkat keyakinan ──
    if conf_pct >= 90:   tingkat_keyakinan = "sangat tinggi"
    elif conf_pct >= 75: tingkat_keyakinan = "tinggi"
    elif conf_pct >= 60: tingkat_keyakinan = "cukup"
    else:                tingkat_keyakinan = "relatif rendah"

    # ── Kalimat-kalimat penyusun ──
    kalimat_pembuka = (
        f"Berdasarkan tinjauan model, berita ini diprediksi sebagai **{kelas_pred}** "
        f"dengan confidence score **{conf_pct:.2f}%** — tingkat keyakinan model tergolong *{tingkat_keyakinan}*."
    )

    tok1, val1 = top[0]; tok1 = tok1.strip()
    tok2, val2 = top[1]; tok2 = tok2.strip()
    searah1, searah2 = arah(val1, kelas_pred), arah(val2, kelas_pred)

    if searah1 and searah2:
        kalimat_fitur = (
            f"Keputusan tersebut terutama dipengaruhi oleh fitur **'{tok1}'** dan **'{tok2}'**, "
            f"keduanya dinilai memberikan kontribusi besar yang mengarahkan model ke kategori **{kelas_pred}**."
        )
    elif searah1 and not searah2:
        kalimat_fitur = (
            f"Fitur **'{tok1}'** menjadi pendorong utama ke arah **{kelas_pred}**, "
            f"sementara fitur **'{tok2}'** justru memberikan sinyal berlawanan yang mengarah ke **{kelas_lawan}**."
        )
    elif not searah1 and searah2:
        kalimat_fitur = (
            f"Fitur **'{tok2}'** memperkuat prediksi ke arah **{kelas_pred}**, "
            f"sedangkan fitur **'{tok1}'** memberikan sinyal yang tidak sejalan, mengarah ke **{kelas_lawan}**."
        )
    else:
        kalimat_fitur = (
            f"Menariknya, fitur **'{tok1}'** dan **'{tok2}'** keduanya memberikan sinyal berlawanan dari prediksi utama, "
            f"namun pengaruhnya belum cukup untuk mengubah keputusan model."
        )

    kalimat_kondisional = ""
    if len(top) >= 3:
        tok3, val3 = top[2]; tok3 = tok3.strip()
        if arah(val3, kelas_pred):
            kalimat_kondisional = (
                f"Selain itu, keberadaan fitur **'{tok3}'** turut memperkuat kecenderungan model "
                f"dalam mengklasifikasikan teks ini ke dalam kelas **{kelas_pred}**."
            )
        else:
            kalimat_kondisional = (
                f"Di sisi lain, fitur **'{tok3}'** memberikan sinyal yang tidak sejalan dengan prediksi utama — "
                f"namun pengaruhnya relatif kecil sehingga tidak mengubah hasil keputusan model."
            )

    if kelas_pred == "Hoaks":
        kalimat_penutup = (
            "⚠️ **Rekomendasi:** Secara keseluruhan, pola linguistik dalam teks ini menunjukkan karakteristik yang umumnya "
            "ditemukan pada konten hoaks. Disarankan untuk memverifikasi informasi ini melalui sumber terpercaya sebelum disebarluaskan."
            if conf_pct >= 75 else
            "⚠️ **Rekomendasi:** Model mendeteksi kemungkinan hoaks, namun dengan tingkat keyakinan yang belum terlalu tinggi. "
            "Tetap lakukan verifikasi mandiri terhadap klaim-klaim dalam teks ini."
        )
    else:
        kalimat_penutup = (
            "⚠️ **Rekomendasi:** Secara keseluruhan, pola linguistik dalam teks ini konsisten dengan berita faktual. "
            "Meski demikian, verifikasi tetap dianjurkan untuk memastikan akurasi informasi."
            if conf_pct >= 75 else
            "⚠️ **Rekomendasi:** Model cenderung mengklasifikasikan teks ini sebagai fakta, namun dengan keyakinan yang moderat — "
            "tetap bijak dalam menyimpulkan."
        )

    bagian = [kalimat_pembuka, kalimat_fitur]
    if kalimat_kondisional:
        bagian.append(kalimat_kondisional)
    bagian.append(kalimat_penutup)
    return "\n\n".join(bagian)