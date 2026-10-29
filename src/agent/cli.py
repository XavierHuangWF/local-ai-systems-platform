
from src.agent.harness import AgentHarness
import logging

def main() -> None:
    """Run the interactive local AI agent."""

    # Display successful tool executions in the CLI.
    tool_logger = logging.getLogger("local_ai.tool_execution")
    tool_logger.setLevel(logging.INFO)
    tool_logger.propagate = False

    if not tool_logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("[Tool] %(message)s")
        )
        tool_logger.addHandler(handler)

    agent = AgentHarness()

    print("=== Local AI Agent ===")
    print("Model: Qwen3-8B")
    print("Tool calling: disabled")
    print("Type /help to see available commands.")

    while True:
        try:
            user_input = input("\nYou: ").strip()

        except (KeyboardInterrupt, EOFError):
            print("\nSession ended.")
            break

        if not user_input:
            continue

        # Exit the application.
        if user_input == "/exit":
            print("Session ended.")
            break

        # Display available commands.
        if user_input == "/help":
            print("\nAvailable commands:")
            print("  /tools on      Enable tool calling")
            print("  /tools off     Disable tool calling")
            print("  /tools status  Show current tool mode")
            print("  /reset         Clear conversation history")
            print("  /exit          End the session")
            print("  /rag on        Enable document retrieval")
            print("  /rag off       Disable document retrieval")
            print("  /rag status    Show current RAG mode")
            continue

        # Reset conversation history.
        if user_input == "/reset":
            agent.reset()
            print("Conversation history cleared.")
            continue

        # Enable registered Python tools.
        if user_input == "/tools on":
            if agent.rag_enabled is True:
                print("Disable RAG with /rag off before enabling tools.")
                continue

            agent.set_tool_mode(True)
            print("Tool calling enabled.")
            continue

        # Disable registered Python tools.
        if user_input == "/tools off":
            agent.set_tool_mode(False)
            print("Tool calling disabled.")
            continue

        # Display current tool mode.
        if user_input == "/tools status":
            status = (
                "enabled" if agent.tools_enabled
                else "disabled"
            )
            print(f"Tool calling is {status}.")
            continue


        # Enable document retrieval.
        if user_input == "/rag on":
            if agent.tools_enabled is True:
                print(
                    "Disable tools with /tools off "
                    "before enabling RAG."
                )
                continue

            try:
                agent.set_rag_mode(True)
                print("RAG enabled. Document index ready.")
            except Exception as exc:
                print(
                    f"Failed to enable RAG: "
                    f"{type(exc).__name__}: {exc}"
                )
            continue

        # Disable document retrieval.
        if user_input == "/rag off":
            agent.set_rag_mode(False)
            print("RAG disabled.")
            continue

        # Display current RAG mode.
        if user_input == "/rag status":
            status = (
                "enabled" if agent.rag_enabled
                else "disabled"
            )

            print(f"RAG is {status}.")
            continue


        # Reject unknown commands.
        if user_input.startswith("/"):
            print("Unknown command. Type /help.")
            continue

        # Send ordinary requests to the Agent Harness.
        try:
            response = agent.run(user_input)
            print(f"\nAgent: {response}")

            # Display evidence sources for successful RAG answers.
            if agent.last_rag_sources:
                print("\n=== Retrieved Sources ===")

                for number, chunk in enumerate(
                    agent.last_rag_sources,
                    start=1,
                ):
                    print(f"\n[{number}] {chunk.chunk_id}")
                    print(f"Source: {chunk.source}")
                    print(f"Heading: {chunk.heading}")


        except Exception as exc:
            print(
                f"\nRequest failed: "
                f"{type(exc).__name__}: {exc}"
            )


if __name__ == "__main__":
    main()
