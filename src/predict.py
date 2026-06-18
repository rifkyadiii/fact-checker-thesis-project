import numpy as np

from src.config import MAX_LEN, CHUNK_WORDS


def chunk_text(text: str) -> list:
    """Memotong teks panjang menjadi potongan ±CHUNK_WORDS kata."""
    words = text.split()
    if len(words) <= CHUNK_WORDS:
        return [text]
    return [" ".join(words[i:i + CHUNK_WORDS]) for i in range(0, len(words), CHUNK_WORDS)]


def predict(text: str, pipe) -> tuple:
    """
    Mengembalikan: (label, prob_hoaks, prob_fakta, num_chunks)
    """
    chunks = chunk_text(text)
    hoaks_scores, fakta_scores = [], []

    for chunk in chunks:
        out    = pipe(chunk, truncation=True, max_length=MAX_LEN)[0]
        scores = {item["label"]: item["score"] for item in out}
        hoaks_scores.append(scores.get("LABEL_1", 0.0))
        fakta_scores.append(scores.get("LABEL_0", 0.0))

    prob_hoaks = float(np.mean(hoaks_scores))
    prob_fakta = float(np.mean(fakta_scores))
    label      = "Hoaks" if prob_hoaks >= prob_fakta else "Fakta"
    return label, prob_hoaks, prob_fakta, len(chunks)