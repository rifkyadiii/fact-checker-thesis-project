import trafilatura

def scrape_article(url: str) -> tuple:
    """
    Mengambil teks narasi murni dari URL menggunakan trafilatura.
    trafilatura secara otomatis mendeteksi konten utama artikel
    dan membuang nav, sidebar, footer, iklan, dll.

    Returns:
        (text: str, error: str)  — salah satu pasti kosong.
    """
    try:
        downloaded = trafilatura.fetch_url(url.strip())

        if not downloaded:
            return "", "Gagal mengunduh halaman. Periksa URL atau koneksi internet."

        text = trafilatura.extract(
            downloaded,
            include_comments=False,   # buang komentar pembaca
            include_tables=False,     # buang tabel (biasanya bukan narasi)
            no_fallback=False,        # aktifkan fallback jika metode utama gagal
            favor_recall=True,        # lebih baik dapat lebih banyak teks daripada terlalu ketat
        )

        if not text or len(text.strip()) < 50:
            return "", "Tidak dapat mengekstrak teks dari halaman ini. Coba salin teks secara manual."

        return text.strip(), ""

    except Exception as e:
        return "", f"Terjadi kesalahan: {str(e)}"