
from src.skills.skill_loader import SkillLoader


def main() -> None:
    print("=== Skill Loader Verification ===")

    loader = SkillLoader()

    skill_path = loader.skills_dir / "research_summary" / "SKILL.md"
    content = skill_path.read_text(encoding="utf-8")

    print("File being loaded:", skill_path)
    print("First 80 characters:", repr(content[:80]))

    # Test 1: Load an existing skill.
    print("\n=== Test 1: Load Research Summary Skill ===")

    skill = loader.load("research_summary")

    print(f"Skill name: {skill.name}")
    print(f"Description: {skill.description}")
    print(f"Instruction length: {len(skill.instructions)} characters")

    assert skill.name == "research_summary"
    assert skill.description
    assert "## Instructions" in skill.instructions

    print("PASS: Research summary skill loaded.")

    # Test 2: Reject an invalid skill name.
    print("\n=== Test 2: Reject Invalid Skill Name ===")

    try:
        loader.load("../research_summary")
    except ValueError as exc:
        print(f"Expected error: {exc}")
        print("PASS: Invalid skill name rejected.")
    else:
        raise AssertionError("Invalid skill name was accepted.")

    # Test 3: Reject a nonexistent skill.
    print("\n=== Test 3: Missing Skill ===")

    try:
        loader.load("nonexistent_skill")
    except FileNotFoundError as exc:
        print(f"Expected error: {exc}")
        print("PASS: Missing skill handled correctly.")
    else:
        raise AssertionError("Missing skill was accepted.")

    print("\n=== Final Result ===")
    print("PASS: All skill loader checks passed.")


if __name__ == "__main__":
    main()
