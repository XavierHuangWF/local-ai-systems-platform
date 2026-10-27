
from src.client.vllm_client import VLLMClient
from src.agent.token_counter import TokenCounter
from src.skills.skill_loader import Skill, SkillLoader

from src.tools.executor import ToolExecutor
from src.tools.definitions import SUMMARIZE_MEASUREMENTS_DEFINITION
from src.tools.statistics_tool import summarize_measurements
from src.agent.tool_loop import run_tool_turn

from src.agent.response_formatter import clean_model_response

SYSTEM_PROMPT = (
    "You are a concise AI systems assistant. "
    "Answer technical questions clearly and accurately."
)

MAX_HISTORY_TURNS = 6


class AgentHarness:
    def __init__(self):
        self.model = VLLMClient()
        self.token_counter = TokenCounter()
        self.skill_loader = SkillLoader()

        self.tool_executor = ToolExecutor()

        self.tool_executor.register(
            definition=SUMMARIZE_MEASUREMENTS_DEFINITION,
            handler=summarize_measurements,
        )

        # Preserve existing chat behavior until tool mode is enabled.
        self.tools_enabled = False

        self.active_skill: Skill | None = None

        self.messages: list[dict[str, str]] = []
        self.reset()

    def _system_content(self) -> str:
        if self.active_skill is None:
            return SYSTEM_PROMPT

        return (
            f"{SYSTEM_PROMPT}\n\n"
            f"Active skill: {self.active_skill.name}\n\n"
            f"{self.active_skill.instructions}"
        )

    def activate_skill(self, skill_name: str) -> None:
        # Load and validate before changing the active skill.
        skill = self.skill_loader.load(skill_name)

        self.active_skill = skill

        # Start a new conversation using the selected skill.
        self.reset()

    def deactivate_skill(self) -> None:
        self.active_skill = None
        self.reset()

    def set_tool_mode(self, enabled: bool) -> None:
        """Enable or disable tool calling for subsequent requests."""

        if not isinstance(enabled, bool):
            raise TypeError("Tool mode must be a boolean.")

        self.tools_enabled = enabled

    def run(self, user_input: str) -> str:
        user_input = user_input.strip()

        if not user_input:
            raise ValueError("User input cannot be empty.")

        recent_history = self.messages[1:][
            -2 * (MAX_HISTORY_TURNS - 1):
        ]

        pending_messages = [
            self.messages[0],
            *recent_history,
            {
                "role": "user",
                "content": user_input,
            },
        ]

        # Remove older exchanges until the input fits.
        while not self.token_counter.fits(pending_messages):
            if len(pending_messages) == 2:
                raise ValueError(
                    "Input exceeds the available token budget."
                )

            del pending_messages[1:3]

        if self.tools_enabled:
            response = run_tool_turn(
                client=self.model.client,
                model=self.model.model,
                messages=pending_messages,
                executor=self.tool_executor,
            )
        else:
            response = self.model.chat(pending_messages)

        # Remove Qwen3 thinking blocks before storing or returning.
        response = clean_model_response(response)

        # Update history only after successful inference.
        self.messages = [
            *pending_messages,
            {
                "role": "assistant",
                "content": response,
            },
        ]

        return response

    def reset(self) -> None:
        self.messages = [
            {
                "role": "system",
                "content": self._system_content(),
            }
        ]
