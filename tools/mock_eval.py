"""Run the repo's evaluation script on the mock predictions with a stubbed judge.

Only the LLM-as-judge call is stubbed (it costs money); the F1 computation and the
result-parsing code run for real.
"""

import sys

sys.argv = ["evaluate_reasoning.py", "--data", "locomo", "--model", "gemini",
            "--file", "mock", "--allfile"]

import eval.evaluate_reasoning as ev


def stub_judge(question, reference, prediction):
    # deterministic stand-in: "correct" when the mock answer is non-empty
    return 1 if str(prediction).strip() else 0


ev.evaluate_llm_judge = stub_judge

if __name__ == "__main__":
    ev.main()
