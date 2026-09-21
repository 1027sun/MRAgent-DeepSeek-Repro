"""Offline end-to-end smoke test for MRAgent.

Runs the real pipeline (rewrite -> embed -> keyword -> graph store -> retrieval ->
tool-calling answer -> result jsonl) for one LoCoMo sample, replacing the OpenRouter
chat and embedding calls with deterministic local stubs. No API key, no network:
this only proves the code paths work on this machine.

Usage from the repo root inside the venv:  python mock_e2e.py [sample_number]
"""

import json
import os
import random
import re
import sys

SAMPLE = sys.argv[1] if len(sys.argv) > 1 else "26"

# config.py parses argv at import time, so build it before importing the repo
sys.argv = ["run.py", "--data", "locomo", "--model", "gemini", "--file", "mock", "--sample", SAMPLE]

import numpy as np  # noqa: E402

import data.embed_rewrite as embed_rewrite  # noqa: E402
import llm.controller as llm_controller  # noqa: E402
from common import config  # noqa: E402

random.seed(0)
np.random.seed(0)

DIA_RE = re.compile(r"dia_id\s*:\s*(D\d+:\d+)")
ID_RE = re.compile(r"(D\d+:\d+)")
ALL_IDS = []
STATS = {"rewrite": 0, "keyword": 0, "qkey": 0, "sort": 0, "keyscore": 0, "answer": 0}
MONTHS = {m: i + 1 for i, m in enumerate(
    "January February March April May June July August September October November December".split())}


def ids_in(payload):
    seen, out = set(), []
    for m in ID_RE.finditer(payload):
        if m.group(1) not in seen:
            seen.add(m.group(1))
            out.append(m.group(1))
    return out


def mock_chat_text(self, *, messages, **kwargs):
    system = messages[0]["content"]
    user = messages[1]["content"] if len(messages) > 1 else ""

    if system.startswith("You are a dialogue processor"):
        STATS["rewrite"] += 1
        text = json.loads(user[user.index("<<<") + 3:user.rindex(">>>")])
        conv_time = "2023-05-08"
        for line in str(text).splitlines():
            if line.startswith("time:"):
                m = re.search(r"(\d{1,2})\s+(\w+),?\s+(\d{4})", line)
                if m:
                    conv_time = "%s-%02d-%02d" % (m.group(3), MONTHS.get(m.group(2), 5), int(m.group(1)))
                break
        sentence = []
        for dia in DIA_RE.findall(str(text)):
            sentence.append({
                "id": dia + "-1", "text": "Mock rewritten sentence for " + dia + ".",
                "tag": "Mock Tag", "origin": dia, "topic": ["t1"], "time": conv_time,
            })
            ALL_IDS.append(dia)
        return {
            "conversation_time": conv_time,
            "sentence": sentence,
            "topics": {"t1": "Mock topic covering the session"},
            "personal_sentences": [{
                "id": "p1", "text": "Mock personal fact.", "tag": "Mock Tag",
                "origin": sentence[0]["origin"] if sentence else "D1:1", "person": "Mock Person",
            }],
        }

    if system.startswith("You are an information extraction system"):
        STATS["keyword"] += 1
        payload = json.loads(user[user.index("<<<") + 3:user.rindex(">>>")])
        return {"sentence": [{"sentence_id": s["id"], "keyword": ["mock", "keyword", "person"]}
                             for s in payload if isinstance(s, dict) and s.get("id")]}

    if system.startswith("You are a keyword extractor"):
        STATS["qkey"] += 1
        question = user[user.index("<<<") + 3:user.rindex(">>>")]
        words = [w for w in (w.strip("?.,!") for w in re.findall(r"[A-Za-z']+", question)) if len(w) > 2]
        words = (words[:4] or []) + ["mock"]  # "mock" is present in every stored sentence
        return {"question_time": "", "keywords": [{"id": w, "alternatives": [w.lower()]} for w in words]}

    if "select at most 20 relevant events" in system:
        STATS["sort"] += 1
        return {"mode": "sort", "events": ids_in(user)[:20]}

    if "produce a relevance score" in system:
        STATS["sort"] += 1
        return {"mode": "score", "relevance_scores": {i: 0.9 for i in ids_in(user)[:20]}}

    if system.startswith("You are going to answer a question with keyword"):
        STATS["keyscore"] += 1
        tags = re.findall(r'"([^"]+)"(?=,\s*"|\s*\])', user)
        return {"keyword": "mock", "tag_scores": {t: 0.9 for t in tags[:5]}}

    raise AssertionError("unmocked system prompt: " + system[:120])


def mock_chat_with_tools_once(self, **kwargs):
    STATS["answer"] += 1
    return "mock answer", (ALL_IDS[:2] or ["D1:1"])


def mock_get_embeddings(inputs, mode="context"):
    arr = np.random.RandomState(len(inputs)).randn(len(inputs), 64).astype("float32")
    arr /= np.linalg.norm(arr, axis=-1, keepdims=True)
    return arr


llm_controller.LLM.chat_text = mock_chat_text
llm_controller.LLM.chat_with_tools_once = mock_chat_with_tools_once
embed_rewrite.get_embeddings = mock_get_embeddings

import run  # noqa: E402

if __name__ == "__main__":
    print("mock E2E: dataset=%s sample=%s result_tag=%s" % (config.dataset, config.sample_id, config.ADDITIONAL_RE))
    run.main()

    sample_id = "conv-" + SAMPLE
    result = config.result_template.format(dataset=config.dataset, sample_id=sample_id)
    print("\n--- artifacts ---")
    for path in (
        config.rewrite_template.format(dataset=config.dataset, sample_id=sample_id),
        config.keyword_template.format(dataset=config.dataset, sample_id=sample_id),
        config.embedding_template.format(dataset=config.dataset, sample_id=sample_id),
        result,
    ):
        size = os.path.getsize(path) if os.path.exists(path) else None
        print(("OK      " if size is not None else "MISSING ") + path, size if size is not None else "")

    if os.path.exists(result):
        with open(result, encoding="utf-8") as fh:
            rows = [json.loads(line) for line in fh if line.strip()]
        print("\nresult rows:", len(rows))
        if rows:
            print("first row:", json.dumps(rows[0], ensure_ascii=False)[:400])

    print("\nstub call counts:", STATS)
