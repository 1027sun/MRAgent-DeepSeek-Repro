"""Load the bundled LoCoMo dataset through the repo's own loader and sanity-check it."""
import json
import os
import sys
import time
from collections import Counter

import numpy as np

from data.get_data import get_data

if __name__ == "__main__":
    t0 = time.time()
    conv, qa, raw_conv, raw_text = get_data("locomo", "data/dataset_locomo.json")

    print(f"samples: {len(conv)}  ({time.time() - t0:.1f}s)")
    total_q = sum(len(v) for v in qa.values())
    total_sessions = sum(len(v) for v in conv.values())
    print(f"sessions: {total_sessions}   questions: {total_q}")

    cats = Counter()
    for sid, items in qa.items():
        for item in items:
            cats[item.get("category")] += 1
    print("question categories:", dict(sorted(cats.items(), key=lambda kv: str(kv[0]))))

    sid = sorted(conv)[0]
    lengths = [len(v) for v in conv[sid].values()]
    print(f"\nsample {sid}: {len(lengths)} sessions, session text lengths "
          f"min={min(lengths)} max={max(lengths)} mean={sum(lengths)/len(lengths):.0f} chars")
    print("first 300 chars of session 1:")
    print(list(conv[sid].values())[0][:300])

    missing = [k for k, v in qa.items() if not v]
    print("\nempty question lists:", missing or "none")
    print("wrote:", [p for p in ("data/conversation_list_locomo.json",
                                 "data/question_list_locomo.json") if os.path.exists(p)])

    # quick numeric sanity check used by the retrieval code
    v = np.random.RandomState(0).randn(4, 8).astype("float32")
    print("torch-style L2 row norms:", np.linalg.norm(v / np.linalg.norm(v, axis=-1, keepdims=True), axis=-1))
    print(f"total {time.time() - t0:.1f}s")
