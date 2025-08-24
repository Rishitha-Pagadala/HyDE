# app.py
import streamlit as st
from pathlib import Path
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from huggingface_hub import InferenceClient

st.set_page_config(page_title="HyDE RAG", page_icon="🧠", layout="centered")

HERE = Path(__file__).parent
INDEX_DIR = HERE / "faiss_index"
IDX_PATH = INDEX_DIR / "index.faiss"
META_PATH = INDEX_DIR / "index_meta.pkl"

EMBED_MODEL_NAME = "sentence-transformers/sentence-t5-large"
K = 3
MIN_SCORE = 0.25  # cosine similarity threshold (tune if needed)
HYPO_TOKENS = 128
FINAL_TOKENS = 256


@st.cache_resource(show_spinner=False)
def load_embedder():
    return SentenceTransformer(EMBED_MODEL_NAME)


@st.cache_resource(show_spinner=False)
def load_faiss_index():
    if not IDX_PATH.exists() or not META_PATH.exists():
        raise RuntimeError(
            "faiss_index not found. Run build_index.py locally and commit 'faiss_index/' to the repo."
        )
    index = faiss.read_index(str(IDX_PATH))
    with open(META_PATH, "rb") as f:
        metadata = pickle.load(f)
    return index, metadata


@st.cache_resource(show_spinner=False)
def get_hf_client():
    token = None
    if "HF_TOKEN" in st.secrets:
        token = st.secrets["HF_TOKEN"]
    else:
        token = st.session_state.get("HF_TOKEN", None)
    if not token:
        st.error("HF token not found. Add HF_TOKEN to Streamlit secrets.")
        st.stop()
    return InferenceClient(model="google/gemma-2-9b", token=token)


def embed_text(text: str, embedder: SentenceTransformer):
    vec = embedder.encode([text], convert_to_numpy=True)
    vec = vec / np.linalg.norm(vec, axis=1, keepdims=True)  # normalize
    return vec.astype("float32")


def search(index, meta, qvec, top_k=K):
    D, I = index.search(qvec, top_k)
    results = []
    for score, idx in zip(D[0], I[0]):
        if idx < 0:
            continue
        results.append((meta[idx]["text"], meta[idx]["source"], float(score)))
    return results


def make_prompt(context_chunks, question):
    """
    Final answer prompt: uses retrieved context ONLY.
    """
    context_text = "\n\n---\n\n".join([c for c in context_chunks])
    prompt = f"""You are a helpful assistant. Answer the user's question STRICTLY using only the CONTEXT below.
If the answer is not contained in the context, reply exactly: "The context does not provide this information."

CONTEXT:
{context_text}

QUESTION:
{question}

ANSWER:"""
    return prompt


def generate_hypothetical_doc(client, question: str) -> str:
    """Generate a concise hypothetical answer (HyDE) to get an embedding for retrieval."""
    hypo_prompt = f"""Generate a short hypothetical answer (1-3 sentences) to the following question.
This is a hypothetical answer used only to find relevant documents via embedding. It does not need to be factually correct.

QUESTION: {question}

HYPOTHETICAL ANSWER:"""
    try:
        resp = client.text_generation(
            hypo_prompt, max_new_tokens=HYPO_TOKENS, do_sample=True, temperature=0.8
        )
    except Exception as e:
        st.error(f"Error generating hypothetical document: {e}")
        # fallback - use the raw question text for embedding
        return question

    # parse response robustly
    if isinstance(resp, dict):
        text = resp.get("generated_text") or resp.get("text") or str(resp)
    elif isinstance(resp, list) and len(resp) > 0:
        first = resp[0]
        if isinstance(first, dict):
            text = first.get("generated_text") or first.get("text") or str(first)
        else:
            text = str(first)
    else:
        text = str(resp)

    return text.strip()


# ----------------- UI -------------------
st.title("🧠 HyDE RAG")

# store conversation history
if "history" not in st.session_state:
    # each entry will be: {"question":..., "hypo":..., "answer":..., "used_contexts":[(src,score)...]}
    st.session_state.history = []

query = st.text_input("Ask a question about the PDF content", "")

if st.button("Ask") and query.strip():
    with st.spinner("Loading resources..."):
        embedder = load_embedder()
        index, metadata = load_faiss_index()
        client = get_hf_client()

    # ---------- STEP 1: Generate hypothetical document ----------
    with st.spinner("Generating hypothetical answer (HyDE)..."):
        hypo_doc = generate_hypothetical_doc(client, query)

    # show the hypothetical doc immediately (for transparency)
    st.markdown("### Hypothetical (HyDE) document — used for retrieval")
    st.write(hypo_doc)

    # ---------- STEP 2: Embed hypothetical doc ----------
    qvec = embed_text(hypo_doc, embedder)

    # ---------- STEP 3: Retrieve from FAISS using hypo embedding ----------
    results = search(index, metadata, qvec, top_k=K)
    # filter by MIN_SCORE; if none pass, fallback to top-k results (so user gets answers)
    filtered = [r for r in results if r[2] >= MIN_SCORE]
    used_fallback = False
    if not filtered and results:
        filtered = results
        used_fallback = True

    # show retrieved sources and scores
    st.markdown("### Retrieved sources (score)")
    if not filtered:
        st.warning("No retrieved documents found.")
        used_contexts = []
    else:
        used_contexts = []
        for i, (txt, src, sc) in enumerate(filtered, start=1):
            st.write(f"{i}. {src} — score: {sc:.4f}")
            # show a short preview
            preview = txt[:800] + ("..." if len(txt) > 800 else "")
            st.caption(preview)
            used_contexts.append((src, sc))

        if used_fallback:
            st.info("No chunk passed the MIN_SCORE threshold; using top-k results as a fallback.")

    # ---------- STEP 4: Final grounded answer using retrieved contexts ----------
    if not filtered:
        # nothing to ground on
        final_answer = "The context does not provide this information."
    else:
        context_chunks = [txt for txt, _, _ in filtered]
        final_prompt = make_prompt(context_chunks, query)

        with st.spinner("Generating final grounded answer..."):
            try:
                resp = client.text_generation(
                    final_prompt, max_new_tokens=FINAL_TOKENS, do_sample=False, temperature=0.2
                )
            except Exception as e:
                st.error(f"Error from HF Inference API: {e}")
                raise

            # parse response robustly
            if isinstance(resp, dict):
                final_answer = resp.get("generated_text") or resp.get("text") or str(resp)
            elif isinstance(resp, list) and len(resp) > 0:
                first = resp[0]
                if isinstance(first, dict):
                    final_answer = first.get("generated_text") or first.get("text") or str(first)
                else:
                    final_answer = str(first)
            else:
                final_answer = str(resp)

            final_answer = final_answer.strip()

            # if model returned empty or obviously not helpful, fallback message
            if not final_answer:
                final_answer = "The context does not provide this information."

    # ---------- Display latest (hypo already shown) ----------
    st.markdown("### Latest Answer")
    st.markdown(f"**Q:** {query}")
    st.markdown(f"**A (grounded):** {final_answer}")

    # ---------- Save into history AFTER showing ----------
    st.session_state.history.append(
        {
            "question": query,
            "hypo": hypo_doc,
            "answer": final_answer,
            "used_contexts": used_contexts,
        }
    )

# ----------------- Show conversation history (previous pairs only) -------------------
if st.session_state.history:
    st.markdown("---")
    st.markdown("## Conversation History (previous Q/A)")
    # show all except the latest — latest is above
    for i, turn in enumerate(st.session_state.history[:-1], 1):
        st.markdown(f"**Q{i}:** {turn['question']}")
        st.markdown(f"**A{i}:** {turn['answer']}")
        # optionally allow viewing the HyDE doc that was generated for that question
        with st.expander("Show HyDE (hypothetical) doc for this question"):
            st.write(turn["hypo"])
        # optionally show which contexts were used
        if turn.get("used_contexts"):
            with st.expander("Contexts used (source — score)"):
                for src, sc in turn["used_contexts"]:
                    st.write(f"- {src} — score: {sc:.4f}")
        st.markdown("---")
