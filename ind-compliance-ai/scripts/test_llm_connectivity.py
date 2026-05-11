from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from llm.llm_client import LLMClient, LLMConfig  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Smoke-test LLM connectivity for Ollama-compatible or OpenAI-compatible backends."
    )
    parser.add_argument("--provider", default="ollama_cloud")
    parser.add_argument("--model", default="gpt-oss:120b-cloud")
    parser.add_argument("--base-url", default="")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument(
        "--prompt",
        default="Reply with exactly: LLM_CONNECTIVITY_OK",
    )
    parser.add_argument(
        "--system-prompt",
        default="Return only the requested answer. No extra text.",
    )
    args = parser.parse_args()

    client = LLMClient(
        LLMConfig(
            provider=args.provider,
            model=args.model,
            base_url=args.base_url,
            api_key=args.api_key,
            request_timeout_s=args.timeout,
        )
    )
    response = client.generate(args.prompt, system_prompt=args.system_prompt)
    print(response)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
