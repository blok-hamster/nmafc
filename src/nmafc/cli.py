"""Unified NMAFC CLI — start, init, chat.

Usage:
    nmafc start [--port PORT] [--config CONFIG] [--production]
    nmafc init
    nmafc chat [--config CONFIG] [--llm PROVIDER_MODEL]
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import threading
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="nmafc",
        description="NMAFC — Neuromorphic Memory Architecture for Conversational AI",
    )
    sub = parser.add_subparsers(dest="command")

    # ── start ──
    start_p = sub.add_parser("start", help="Start backend + frontend servers")
    start_p.add_argument("--port", type=int, default=8000, help="Backend port (default: 8000)")
    start_p.add_argument("--config", default="configs/default.toml", help="Config TOML path")
    start_p.add_argument("--production", action="store_true", help="Build frontend and serve from one port")
    start_p.add_argument("--host", default="0.0.0.0", help="Bind address (default: 0.0.0.0)")

    # ── init ──
    sub.add_parser("init", help="Interactive setup wizard")

    # ── chat ──
    chat_p = sub.add_parser("chat", help="Interactive terminal chat")
    chat_p.add_argument("--config", default="configs/default.toml", help="Config TOML path")
    chat_p.add_argument("--llm", default=None, help="Override LLM provider/model")
    chat_p.add_argument("--embedding", default=None, help="Override embedding provider/model")

    args = parser.parse_args()

    if args.command == "start":
        cmd_start(args)
    elif args.command == "init":
        cmd_init()
    elif args.command == "chat":
        cmd_chat(args)
    else:
        parser.print_help()


# ═══════════════════════════════════════════════════════════════════
#  nmafc start
# ═══════════════════════════════════════════════════════════════════

def cmd_start(args: argparse.Namespace) -> None:
    """Start uvicorn backend + Next.js frontend."""
    try:
        from rich.console import Console
        console = Console()
        _print = console.print
    except ImportError:
        console = None
        _print = print

    config_path = Path(args.config)
    if not config_path.exists():
        _print(f"[red]Config not found:[/red] {config_path}")
        sys.exit(1)

    web_ui_dir = Path(__file__).resolve().parent.parent.parent / "web-ui"
    if not web_ui_dir.exists():
        _print(f"[yellow]web-ui/ not found at {web_ui_dir}[/yellow]")
        _print("Running backend only.")
        web_ui_dir = None

    backend_port = args.port
    frontend_port = backend_port + 1
    env = os.environ.copy()
    env["NMAFC_CONFIG_PATH"] = str(config_path)

    procs: list[subprocess.Popen] = []

    def shutdown(*_a: object) -> None:
        _print("\n[yellow]Shutting down...[/yellow]")
        for p in procs:
            try:
                p.terminate()
                p.wait(timeout=5)
            except Exception:
                p.kill()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    if args.production:
        _print("[bold]Production mode[/bold] — building frontend...")
        if web_ui_dir:
            build = subprocess.run(
                ["npm", "run", "build"],
                cwd=str(web_ui_dir),
                env=env,
            )
            if build.returncode != 0:
                _print("[red]Frontend build failed[/red]")
                sys.exit(1)
            _print("[green]Frontend built successfully[/green]")

        _print(f"[green]Starting server on {args.host}:{backend_port}[/green]")
        proc = subprocess.Popen(
            [
                sys.executable, "-m", "uvicorn",
                "nmafc.web.app:create_app",
                "--factory",
                "--host", args.host,
                "--port", str(backend_port),
            ],
            env=env,
        )
        procs.append(proc)
        proc.wait()

    else:
        # Dev mode: start both servers
        _print(f"[bold]Dev mode[/bold]")
        _print(f"  Backend:  [cyan]http://localhost:{backend_port}[/cyan]")
        if web_ui_dir:
            _print(f"  Frontend: [cyan]http://localhost:{frontend_port}[/cyan]")
        _print()

        # Start backend
        backend_proc = subprocess.Popen(
            [
                sys.executable, "-m", "uvicorn",
                "nmafc.web.app:create_app",
                "--factory",
                "--host", "127.0.0.1",
                "--port", str(backend_port),
                "--reload",
            ],
            env=env,
        )
        procs.append(backend_proc)

        # Start frontend
        frontend_proc = None
        if web_ui_dir:
            frontend_env = env.copy()
            frontend_env["PORT"] = str(frontend_port)
            frontend_env["NEXT_PUBLIC_API_URL"] = f"http://localhost:{backend_port}"
            frontend_proc = subprocess.Popen(
                ["npm", "run", "dev"],
                cwd=str(web_ui_dir),
                env=frontend_env,
            )
            procs.append(frontend_proc)

        # Wait for either to exit
        try:
            while True:
                for p in procs:
                    if p.poll() is not None:
                        shutdown()
                import time
                time.sleep(1)
        except KeyboardInterrupt:
            shutdown()


# ═══════════════════════════════════════════════════════════════════
#  nmafc init
# ═══════════════════════════════════════════════════════════════════

PROVIDER_OPTIONS = [
    ("openai/gpt-4o-mini", "OpenAI GPT-4o Mini (default)"),
    ("openai/gpt-4o", "OpenAI GPT-4o"),
    ("anthropic/claude-sonnet-4-20250514", "Anthropic Claude Sonnet 4"),
    ("anthropic/claude-haiku-4-5-20251001", "Anthropic Claude Haiku 4.5"),
    ("groq/llama-3.1-70b-versatile", "Groq Llama 3.1 70B"),
    ("groq/llama-3.1-8b-instant", "Groq Llama 3.1 8B (fast)"),
    ("ollama/llama3", "Ollama Llama 3 (local)"),
    ("together/meta-llama/Llama-3-70b-chat-hf", "Together Llama 3 70B"),
]

EMBEDDING_OPTIONS = [
    ("openai/text-embedding-3-small", "OpenAI text-embedding-3-small (default)"),
    ("openai/text-embedding-3-large", "OpenAI text-embedding-3-large"),
    ("ollama/nomic-embed-text", "Ollama nomic-embed-text (local)"),
    ("fastembed/BAAI/bge-small-en-v1.5", "FastEmbed BGE Small (local, no API key)"),
]


def cmd_init() -> None:
    """Interactive setup wizard."""
    try:
        from rich.console import Console
        from rich.panel import Panel
        from rich.prompt import Prompt, Confirm
        console = Console()
    except ImportError:
        sys.exit("Install rich for the setup wizard: pip install 'nmafc[cli]'")

    console.print(Panel.fit(
        "[bold]NMAFC Setup Wizard[/bold]\n\n"
        "Configure your LLM provider, embedding model,\n"
        "and storage paths.",
        border_style="cyan",
    ))

    # ── LLM ──
    console.print("\n[bold]LLM Provider[/bold]")
    for i, (_, label) in enumerate(PROVIDER_OPTIONS, 1):
        console.print(f"  {i}. {label}")
    choice = Prompt.ask(
        "Select LLM provider",
        default="1",
        choices=[str(i) for i in range(1, len(PROVIDER_OPTIONS) + 1)],
    )
    llm_model = PROVIDER_OPTIONS[int(choice) - 1][0]

    # ── Embedding ──
    console.print("\n[bold]Embedding Model[/bold]")
    for i, (_, label) in enumerate(EMBEDDING_OPTIONS, 1):
        console.print(f"  {i}. {label}")
    choice = Prompt.ask(
        "Select embedding model",
        default="1",
        choices=[str(i) for i in range(1, len(EMBEDDING_OPTIONS) + 1)],
    )
    embed_model = EMBEDDING_OPTIONS[int(choice) - 1][0]

    # ── API Keys ──
    env_lines: list[str] = []
    console.print("\n[bold]API Keys[/bold]")

    providers_needing_keys = set()
    for key_model in [llm_model, embed_model]:
        provider = key_model.split("/")[0]
        if provider in ("openai", "anthropic", "groq", "openrouter", "together"):
            providers_needing_keys.add(provider)

    key_env_map = {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "groq": "GROQ_API_KEY",
        "openrouter": "OPENROUTER_API_KEY",
        "together": "TOGETHER_API_KEY",
    }

    for provider in sorted(providers_needing_keys):
        env_var = key_env_map[provider]
        existing = os.environ.get(env_var, "")
        if existing:
            console.print(f"  [green]{env_var}[/green] already set")
        else:
            key = Prompt.ask(f"  {env_var}", default="")
            if key:
                env_lines.append(f"{env_var}={key}")

    # ── Storage ──
    console.print("\n[bold]Storage[/bold]")
    data_dir = Prompt.ask("Data directory", default="./data")

    # ── Write .env ──
    env_path = Path(".env")
    existing_env = ""
    if env_path.exists():
        existing_env = env_path.read_text()

    new_lines = []
    for line in env_lines:
        if not any(existing.startswith(line.split("=")[0] + "=") for existing in existing_env.splitlines()):
            new_lines.append(line)

    new_lines.extend([
        f"NMAFC_EMBEDDING_DIM=1536",
    ])

    if new_lines:
        with open(env_path, "a") as f:
            if existing_env and not existing_env.endswith("\n"):
                f.write("\n")
            for line in new_lines:
                f.write(line + "\n")
        console.print(f"\n[green]Updated {env_path}[/green]")

    # ── Write config TOML ──
    config_dir = Path("configs")
    config_dir.mkdir(exist_ok=True)
    config_path = config_dir / "custom.toml"

    config_content = f"""# NMAFC Configuration — generated by `nmafc init`

[storage]
hot_uri = "{data_dir}/lancedb"
cold_uri = "{data_dir}/cold.db"
event_log_uri = "{data_dir}/events.db"

[decay]
lambda_core_anchor = 0.0
lambda_active_context = 0.05
lambda_ephemeral = 0.69
eta = 0.15
gamma = 0.1
w_prune = 0.1

[retrieval]
theta = 0.45
top_k = 10
fallback_keyword_limit = 20

[time]
unit = "turns"

[embedding]
provider_model = "{embed_model}"
dim = 1536

[llm]
provider_model = "{llm_model}"
"""
    config_path.write_text(config_content)
    console.print(f"[green]Wrote {config_path}[/green]")

    # ── Done ──
    console.print(Panel.fit(
        "[bold green]Setup complete![/bold green]\n\n"
        f"  LLM:      [cyan]{llm_model}[/cyan]\n"
        f"  Embedding: [cyan]{embed_model}[/cyan]\n"
        f"  Config:    [cyan]{config_path}[/cyan]\n\n"
        "Next steps:\n"
        "  [bold]nmafc start[/bold]              — launch backend + frontend\n"
        "  [bold]nmafc chat[/bold]               — interactive terminal chat\n"
        "  [bold]nmafc chat --llm ollama/llama3[/bold] — use a different model",
        border_style="green",
    ))


# ═══════════════════════════════════════════════════════════════════
#  nmafc chat
# ═══════════════════════════════════════════════════════════════════

def cmd_chat(args: argparse.Namespace) -> None:
    """Interactive terminal chat REPL."""
    try:
        from rich.console import Console
        from rich.markdown import Markdown
        from rich.panel import Panel
        from rich.table import Table
        console = Console()
    except ImportError:
        sys.exit("Install rich for terminal chat: pip install 'nmafc[cli]'")

    # Load .env so env-var overrides reach from_env_or_toml()
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    # Load config
    from nmafc.storage.config import NMafcConfig

    config_path = Path(args.config)
    if config_path.exists():
        config = NMafcConfig.from_env_or_toml(config_path)
    else:
        config = NMafcConfig.from_env_or_toml()

    # Override providers from CLI flags
    if args.llm:
        config.llm_provider_model = args.llm
    if args.embedding:
        config.embedding_provider_model = args.embedding

    console.print(Panel.fit(
        "[bold]NMAFC Chat[/bold]\n\n"
        f"  LLM:      [cyan]{config.llm_provider_model}[/cyan]\n"
        f"  Embedding: [cyan]{config.embedding_provider_model}[/cyan]\n\n"
        "Commands: /stats, /memory, /events, /rollback N, /quit",
        border_style="cyan",
    ))

    # Initialize memory
    console.print("[dim]Initializing memory...[/dim]")
    try:
        from nmafc.wrapper import NeuromorphicMemory
        memory = NeuromorphicMemory.from_config(config=config)
    except Exception as e:
        console.print(f"[red]Failed to initialize:[/red] {e}")
        sys.exit(1)

    console.print("[green]Ready![/green]\n")

    history: list[dict] = []

    try:
        while True:
            try:
                user_input = console.input("[bold blue]You:[/bold blue] ").strip()
            except EOFError:
                break

            if not user_input:
                continue

            # Handle commands
            if user_input.startswith("/"):
                _handle_chat_command(user_input, memory, console)
                continue

            history.append({"role": "user", "content": user_input})

            # Process turn
            console.print("[dim]Thinking...[/dim]", end="\r")
            try:
                import asyncio
                response = asyncio.run(
                    memory.process_turn(user_input, history)
                )
            except Exception as e:
                console.print(f"[red]Error:[/red] {e}")
                continue

            history.append({"role": "assistant", "content": response})

            console.print()
            console.print(Panel(
                Markdown(response),
                title=f"[bold]Assistant[/bold] (turn {memory.current_turn})",
                border_style="green",
            ))

            # Show quick stats
            stats = memory.get_hot_stats()
            console.print(
                f"[dim]  Records: {stats['count']} | "
                f"Avg weight: {stats['avg_weight']:.3f} | "
                f"Types: {stats['types']}[/dim]\n"
            )

    except KeyboardInterrupt:
        pass
    finally:
        memory.close()
        console.print("\n[dim]Session ended.[/dim]")


def _handle_chat_command(cmd: str, memory: object, console: object) -> None:
    """Handle slash commands in chat mode."""
    import asyncio

    parts = cmd.split()
    command = parts[0].lower()

    if command == "/quit" or command == "/exit":
        raise SystemExit(0)

    elif command == "/stats":
        stats = memory.get_hot_stats()
        cold = memory.get_cold_stats()
        events = memory.get_event_stats()
        from rich.table import Table
        table = Table(title="System Stats")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        table.add_row("Turn", str(memory.current_turn))
        table.add_row("Hot Records", str(stats["count"]))
        table.add_row("Avg Weight", f"{stats['avg_weight']:.4f}")
        table.add_row("Types", str(stats["types"]))
        table.add_row("Cold Events", str(cold["total_events"]))
        table.add_row("Active Cold", str(cold["active_events"]))
        table.add_row("Cognitive Events", str(events["total_events"]))
        console.print(table)

    elif command == "/memory":
        records = memory._hot.get_all()
        from rich.table import Table
        table = Table(title=f"Hot RAM ({len(records)} records)")
        table.add_column("Entity", style="cyan")
        table.add_column("Type", style="yellow")
        table.add_column("Weight", style="green")
        table.add_column("Consol.", style="dim")
        table.add_column("Fact", max_width=40)
        for r in sorted(records, key=lambda x: x.weight, reverse=True):
            table.add_row(
                r.entity_name, r.memory_type.value,
                f"{r.weight:.4f}", str(r.consolidation_index),
                r.fact_content[:40],
            )
        console.print(table)

    elif command == "/events":
        events = memory.get_events(limit=20)
        from rich.table import Table
        table = Table(title=f"Recent Events (last {len(events)})")
        table.add_column("Turn", style="dim")
        table.add_column("Type", style="cyan")
        table.add_column("Entity", style="yellow")
        table.add_column("Details")
        for ev in reversed(events):
            details = ""
            if ev.old_weight is not None and ev.new_weight is not None:
                details = f"{ev.old_weight:.3f} → {ev.new_weight:.3f}"
            elif ev.suppressed_by:
                details = f"by {ev.suppressed_by}"
            table.add_row(str(ev.turn), ev.event_type.value, ev.entity_name, details)
        console.print(table)

    elif command == "/rollback":
        if len(parts) < 2:
            console.print("[red]Usage: /rollback <turn_number>[/red]")
            return
        try:
            to_turn = int(parts[1])
        except ValueError:
            console.print("[red]Turn must be a number[/red]")
            return
        restored = asyncio.run(memory.rollback(to_turn))
        console.print(f"[green]Restored {restored} records to turn {to_turn}[/green]")

    else:
        console.print(f"[red]Unknown command: {command}[/red]")


if __name__ == "__main__":
    main()
