"""Local chat CLI for PHYSMOL + optional small LLM shell."""

from __future__ import annotations

import argparse
import json

from .agent_bridge import PhysmolAgentBridge, SmallLLMClient


def main():
    parser = argparse.ArgumentParser(description="Chat with PHYSMOL agent bridge")
    parser.add_argument("--vsa-dim", type=int, default=4096)
    parser.add_argument("--broca-checkpoint", default="./checkpoints/broca/model")
    parser.add_argument("--learning-dir", default="./checkpoints/learning")
    parser.add_argument("--trace-dir", default="./checkpoints/evolution")
    parser.add_argument("--no-llm", action="store_true", help="Disable LLM verbalization")
    parser.add_argument("--llm-provider", default="", choices=["", "openai", "ollama"])
    parser.add_argument("--llm-endpoint", default="")
    parser.add_argument("--llm-model", default="")
    parser.add_argument("--llm-api-key", default="")
    parser.add_argument("--once", default="", help="Ask a single question and exit")
    args = parser.parse_args()

    llm_client = None
    if not args.no_llm and args.llm_endpoint and args.llm_model:
        llm_client = SmallLLMClient(
            provider=args.llm_provider or "openai",
            endpoint=args.llm_endpoint,
            model=args.llm_model,
            api_key=args.llm_api_key,
        )

    bridge = PhysmolAgentBridge(
        vsa_dim=args.vsa_dim,
        broca_checkpoint=args.broca_checkpoint,
        learning_dir=args.learning_dir,
        trace_dir=args.trace_dir,
        llm_client=llm_client,
    )

    if args.once:
        print(bridge.chat(args.once, use_llm=not args.no_llm))
        return

    print("PHYSMOL Agent CLI")
    print("Commands: /status, /tools, /save, /export <dir>, /good, /bad <correction>, /quit")
    while True:
        try:
            text = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not text:
            continue
        if text in {"/quit", "quit", "exit"}:
            break
        if text == "/status":
            print(json.dumps(bridge.call_tool("status", {}, record=False), ensure_ascii=False, indent=2))
            continue
        if text == "/tools":
            print(json.dumps(bridge.list_tools(), ensure_ascii=False, indent=2))
            continue
        if text == "/save":
            print(bridge.call_tool("save_learning_state", {}, record=False)["response"])
            continue
        if text.startswith("/export"):
            parts = text.split(maxsplit=1)
            out_dir = parts[1] if len(parts) > 1 else "./checkpoints/evolution/export"
            print(bridge.call_tool("export_training_batch", {"out_dir": out_dir}, record=False)["response"])
            continue
        if text == "/good":
            print(bridge.call_tool("record_feedback", {"feedback": "good"}, record=False)["response"])
            continue
        if text.startswith("/bad"):
            correction = text.split(maxsplit=1)[1] if " " in text else ""
            print(bridge.call_tool(
                "record_feedback",
                {"feedback": "bad", "correction": correction},
                record=False,
            )["response"])
            continue

        response = bridge.chat(text, use_llm=not args.no_llm)
        print(f"PHYSMOL: {response}")

    bridge.call_tool("save_learning_state", {}, record=False)
    print("Saved learning state.")


if __name__ == "__main__":
    main()
