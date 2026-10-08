
from src.agent.harness import AgentHarness


def main() -> None:
    agent = AgentHarness()

    print("Local AI Agent")
    print("Commands: /reset, /exit")

    while True:
        try:
            user_input = input("\nYou> ").strip()

            if not user_input:
                continue

            if user_input.lower() == "/exit":
                print("Exiting.")
                break

            if user_input.lower() == "/reset":
                agent.reset()
                print("Conversation reset.")
                continue

            response = agent.run(user_input)
            print(f"\nAssistant> {response}")

        except KeyboardInterrupt:
            print("\nExiting.")
            break

        except Exception as exc:
            print(f"Error: {exc}")


if __name__ == "__main__":
    main()
