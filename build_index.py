# build_index.py
import os
import pickle
from pathlib import Path
from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from PyPDF2 import PdfReader

# config
HERE = Path(__file__).parent
PDF_PATH = HERE / "x_biology_em.pdf"   # file you provided. Ensure this exists.
OUT_DIR = HERE / "faiss_index"
MODEL_NAME = "sentence-transformers/sentence-t5-large"
CHUNK_SIZE = 1000        # characters per chunk
CHUNK_OVERLAP = 200      # overlap between chunks

def extract_text_from_pdf(pdf_path: Path) -> str:
    reader = PdfReader(str(pdf_path))
    pages = []
    for p in reader.pages:
        try:
            pages.append(p.extract_text() or "")
        except Exception:
            pages.append("")
    return "\n\n".join(pages)

def chunk_text(text: str, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP) -> List[str]:
    text = text.replace("\r\n", "\n")
    chunks = []
    start = 0
    N = len(text)
    while start < N:
        end = min(start + chunk_size, N)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - overlap if end < N else end
    return chunks

def normalize_embeddings(embs: np.ndarray):
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    norms[norms == 0] = 1e-9
    return embs / norms

def main():
    if not PDF_PATH.exists():
        raise FileNotFoundError(f"{PDF_PATH} not found. Place the PDF in the repo root.")

    print("Extracting text from PDF...")
    text = extract_text_from_pdf(PDF_PATH)

    print("Chunking text...")
    chunks = chunk_text(text)
    print(f"Chunks created: {len(chunks)}")

    print("Loading SentenceTransformer...")
    model = SentenceTransformer(MODEL_NAME)

    print("Computing embeddings (this may take minutes locally)...")
    embeddings = model.encode(chunks, show_progress_bar=True, convert_to_numpy=True)
    embeddings = normalize_embeddings(embeddings).astype("float32")

    dim = embeddings.shape[1]
    print(f"Embedding dim: {dim}")

    print("Building FAISS index (IndexFlatIP for cosine similarity on normalized vectors)...")
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    OUT_DIR.mkdir(exist_ok=True)
    idx_path = OUT_DIR / "index.faiss"
    meta_path = OUT_DIR / "index_meta.pkl"

    print(f"Saving FAISS index to {idx_path} and metadata to {meta_path} ...")
    faiss.write_index(index, str(idx_path))

    # Save metadata: original chunks and simple source info
    metadata = [{"text": chunks[i], "source": f"{PDF_PATH.name}#chunk-{i}"} for i in range(len(chunks))]
    with open(meta_path, "wb") as f:
        pickle.dump(metadata, f)

    print("Done. Commit the 'faiss_index' folder to your repo before deploying to Streamlit Cloud.")

if __name__ == "__main__":
    main()
