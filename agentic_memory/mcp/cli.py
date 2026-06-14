"""Zero-dep CLI. Uses argparse and stdlib only."""

import asyncio
import sys
import argparse


async def cmd_serve(args):
    from mem_mcp.server import mcp
    print("Starting Agentic Memory MCP server on stdio...", file=sys.stderr)
    sys.stderr.flush()
    await mcp.run_stdio_async()


async def cmd_chat(args):
    user_id = int(args.user_id) if args.user_id else 1
    print(f"Chat mode for user {user_id}")
    from mem.pipeline import chat
    history = []
    while True:
        try:
            msg = input("> ")
        except (EOFError, KeyboardInterrupt):
            break
        if msg.lower() in ("/exit", "/quit"):
            break
        response = await chat(msg, user_id, history)
        history.append({"role": "user", "content": msg})
        history.append({"role": "assistant", "content": response})
        print(f"\nAI: {response}\n")


async def cmd_init(args):
    print("Memory store initialized.")


def cmd_benchmark(args):
    from benchmark.run import run_benchmark
    run_benchmark()


def main():
    parser = argparse.ArgumentParser(description="Agentic Memory")
    sub = parser.add_subparsers(dest="command")

    serve_p = sub.add_parser("serve", help="Start MCP server")
    chat_p = sub.add_parser("chat", help="Start interactive chat")
    chat_p.add_argument("user_id", nargs="?", default="1", help="User ID")
    sub.add_parser("init-db", help="Initialize store")
    bench_p = sub.add_parser("benchmark", help="Run benchmarks")

    args = parser.parse_args()

    if args.command == "serve":
        asyncio.run(cmd_serve(args))
    elif args.command == "chat":
        asyncio.run(cmd_chat(args))
    elif args.command == "init-db":
        asyncio.run(cmd_init(args))
    elif args.command == "benchmark":
        cmd_benchmark(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
