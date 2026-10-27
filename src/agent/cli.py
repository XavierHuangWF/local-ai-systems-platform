
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
            continue

        # Reset conversation history.
        if user_input == "/reset":
            agent.reset()
            print("Conversation history cleared.")
            continue

        # Enable registered Python tools.
        if user_input == "/tools on":
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

        # Reject unknown commands.
        if user_input.startswith("/"):
            print("Unknown command. Type /help.")
            continue

        # Send ordinary requests to the Agent Harness.
        try:
            response = agent.run(user_input)
            print(f"\nAgent: {response}")

        except Exception as exc:
            print(
                f"\nRequest failed: "
                f"{type(exc).__name__}: {exc}"
            )


if __name__ == "__main__":
    main()
