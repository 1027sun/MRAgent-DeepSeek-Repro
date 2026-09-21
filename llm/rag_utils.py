import os
# avoid torchvision compatibility issues when importing torch
os.environ["TRANSFORMERS_NO_TORCHVISION"] = "1"
os.environ["DISABLE_TORCHVISION"] = "1"
import numpy as np
import torch
from tqdm import tqdm
from dotenv import load_dotenv

from llm.embeddings import get_openai_embedding, set_openai_key

load_dotenv()  # pick up EMBED_BACKEND / HF_HOME / HF_ENDPOINT from .env

# ------------------------------------------------------------------ embedding backend
# "api"   -> OpenAI-compatible /embeddings endpoint (OpenRouter, OpenAI, a relay ...)
# "local" -> sentence-transformers model on CPU: free, offline, no API key
EMBED_BACKEND = os.getenv("EMBED_BACKEND", "api").strip().lower()
LOCAL_EMBED_MODEL = os.getenv("LOCAL_EMBED_MODEL", "BAAI/bge-base-en-v1.5")
LOCAL_EMBED_BATCH = int(os.getenv("LOCAL_EMBED_BATCH", "64"))
LOCAL_EMBED_DEVICE = os.getenv("LOCAL_EMBED_DEVICE", "cpu")
# BGE models are trained with an instruction prefix on the QUERY side only. The `mode`
# argument of get_embeddings() tells the two sides apart, so the prefix goes on queries.
BGE_QUERY_PREFIX = os.getenv(
    "BGE_QUERY_PREFIX", "Represent this sentence for searching relevant passages: "
)

_LOCAL_MODEL = None


def _local_model():
    global _LOCAL_MODEL
    if _LOCAL_MODEL is None:
        from sentence_transformers import SentenceTransformer  # lazy: only for backend=local
        _LOCAL_MODEL = SentenceTransformer(LOCAL_EMBED_MODEL, device=LOCAL_EMBED_DEVICE)
    return _LOCAL_MODEL


def _get_embeddings_local(inputs, mode):
    """CPU sentence-transformers path.

    normalize_embeddings=True is REQUIRED: common/utils.py ranks candidates by raw dot
    product (similarity="dot"), which only equals cosine while rows are unit length.
    """
    model = _local_model()
    texts = [("" if t is None else str(t)) for t in inputs]
    if mode == "query" and BGE_QUERY_PREFIX:
        texts = [BGE_QUERY_PREFIX + t for t in texts]
    vecs = model.encode(
        texts,
        batch_size=LOCAL_EMBED_BATCH,
        normalize_embeddings=True,
        show_progress_bar=len(texts) > LOCAL_EMBED_BATCH,
        convert_to_numpy=True,
    )
    return np.asarray(vecs, dtype="float32")


def get_embeddings(inputs, mode='context'):
    """Sentence/query vectors with L2 normalization (unit-length rows either way).

    Backend is selected by EMBED_BACKEND:
      - "api"   (default): OpenAI-compatible /embeddings endpoint, batched 24 at a time
      - "local":           sentence-transformers model on CPU (free, offline)
    `mode` is 'query' for questions and 'context' for stored sentences; the local BGE
    path uses it to prepend the retrieval instruction to queries only.
    """
    if EMBED_BACKEND == "local":
        return _get_embeddings_local(inputs, mode)

    set_openai_key()
    all_embeddings = []
    batch_size = 24
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    with torch.no_grad():
        for i in tqdm(range(0, len(inputs), batch_size)):
            vecs = get_openai_embedding(inputs[i:(i + batch_size)])
            embeddings = torch.tensor(vecs, dtype=torch.float32, device=device)
            embeddings = torch.nn.functional.normalize(embeddings, dim=-1)
            all_embeddings.append(embeddings)
    return torch.cat(all_embeddings, dim=0).cpu().numpy()
