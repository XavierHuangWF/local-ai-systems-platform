
from dataclasses import dataclass
from pathlib import Path
import re


@dataclass(frozen=True)
class Skill:
    name: str
    description: str
    instructions: str


class SkillLoader:
    def __init__(self, skills_dir: Path | None = None):
        project_root = Path(__file__).resolve().parents[2]

        self.skills_dir = (
            skills_dir or project_root / "skills"
        ).resolve()

    def load(self, skill_name: str) -> Skill:
        # Only allow valid skill directory names.
        if not re.fullmatch(r"[a-z][a-z0-9_]*", skill_name):
            raise ValueError("Invalid skill name.")

        skill_path = (
            self.skills_dir / skill_name / "SKILL.md"
        ).resolve()

        if not skill_path.is_relative_to(self.skills_dir):
            raise ValueError("Skill path is outside skills directory.")

        if not skill_path.is_file():
            raise FileNotFoundError(
                f"Skill not found: {skill_name}"
            )

        content = skill_path.read_text(encoding="utf-8")
        lines = content.splitlines()

        if not lines or lines[0].strip() != "---":
            raise ValueError("Missing skill metadata.")

        end = next(
            (
                i for i, line in enumerate(lines[1:], start=1)
                if line.strip() == "---"
            ),
            None,
        )

        if end is None:
            raise ValueError("Invalid skill metadata format.")

        metadata = {}

        for line in lines[1:end]:
            key, separator, value = line.partition(":")

            if not separator:
                raise ValueError("Invalid metadata field.")

            metadata[key.strip()] = value.strip()

        if metadata.get("name") != skill_name:
            raise ValueError("Skill name does not match directory.")

        description = metadata.get("description", "")
        instructions = "\n".join(lines[end + 1:]).strip()

        if not description or not instructions:
            raise ValueError("Incomplete skill definition.")

        return Skill(
            name=skill_name,
            description=description,
            instructions=instructions,
        )
