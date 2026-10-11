# Skills and Dynamic Context Management

## Objective

Implement a reusable mechanism for loading task-specific instructions into a local Large Language Model (LLM) application. Skill definitions are stored separately from application code so they can be reviewed, modified, and versioned independently.

**Implementation status:** The skill definition and loader have been implemented and verified. Skill selection and injection into `AgentHarness` have not yet been implemented.

## Architecture

```text
skills/research_summary/SKILL.md
               |
               v
        SkillLoader.load()
               |
               v
        Skill data object
        - name
        - description
        - instructions
               |
               v
    Future AgentHarness integration
```

The current loader reads and validates a local file. It does not execute the skill, invoke the model, or change an agent's instructions automatically.

## Implementation

### Skill definition

**File:** `skills/research_summary/SKILL.md`

The file contains front matter with the skill name and description, followed by Markdown instructions for the research-summary task. Its instructions define the expected summary sections and require evidence-based reporting, accurate quantitative results, explicit limitations, and no unsupported citations.

### Skill loader

**File:** `src/skills/skill_loader.py`

The loader:

- Resolves the configured skills directory relative to the project root.
- Validates skill names against an allowed pattern.
- Resolves the requested `SKILL.md` path and checks that it remains inside the skills directory.
- Reads the file and extracts simple `key: value` metadata between two `---` delimiters.
- Requires the metadata name to match the requested skill directory.
- Rejects missing or incomplete definitions with an explicit exception.
- Returns a `Skill` object containing `name`, `description`, and `instructions`.

The metadata parser supports simple fields rather than the complete YAML (YAML Ain't Markup Language) syntax.

### Verification script

**File:** `scripts/verify_skill_loader.py`

From the repository root, with the project's Python environment activated, run:

```bash
python -m scripts.verify_skill_loader
```

This script runs without the vLLM inference server or model execution.

## Verification Results

### Test 1 — Load an existing skill

The script loaded `research_summary`, checked the returned name and description, and verified that its instructions contain the `## Instructions` heading.

```text
Skill name: research_summary
Description: Summarize technical research with evidence and limitations.
Instruction length: 1050 characters
PASS: Research summary skill loaded.
```

### Test 2 — Reject an invalid skill name

The script requested `../research_summary` and confirmed that the loader rejected it.

```text
Expected error: Invalid skill name.
PASS: Invalid skill name rejected.
```

### Test 3 — Handle a missing skill

The script requested `nonexistent_skill` and confirmed that the loader reported a missing file.

```text
Expected error: Skill not found: nonexistent_skill
PASS: Missing skill handled correctly.
```

### Overall result

```text
PASS: All skill loader checks passed.
```

These checks verify local loading and validation behavior. They do not test model responses, malicious skill contents, or skill selection by the agent.

## Debugging Record

### Observed failure

The initial verification failed with:

```text
ValueError: Missing skill metadata.
```

The diagnostic script printed the file path and the first characters of the on-disk skill file using Python's `repr()` function. The content began with `\n---`, revealing a leading newline before the opening front-matter delimiter.

### Root cause and correction

The parser requires `---` to be the first line. Removing the leading blank line from `skills/research_summary/SKILL.md` corrected the formatting error. All three verification checks subsequently passed.

## Engineering Decisions

### Separate instructions from application code

Maintaining skills as Markdown files allows task instructions to change without editing Python source code. Each skill can be independently reviewed and versioned.

### Validate names and resolved paths

Validation rejects malformed skill names and prevents requests from resolving outside the configured skills directory. These checks limit unintended file access; they do not establish that the content of an allowed skill is safe or trustworthy.

### Verify deterministic behavior without inference

Parsing, validation, and file loading do not require a language model. A standalone verification script makes these checks fast and repeatable without using Graphics Processing Unit (GPU) resources.

## Current Limitations

- Only the `research_summary` skill has been implemented and verified.
- Skill activation and inclusion in `AgentHarness` prompts are not yet implemented.
- The front-matter parser does not support the full YAML specification.
- The loader assumes skill files inside the project are trusted inputs; it does not inspect instructions for prompt injection.
- Loading a skill does not execute code or grant tool permissions.
- No Qwen3 behavior evaluation has been performed with the skill instructions.

## Next Implementation Stage

Add explicit skill selection in `AgentHarness`, incorporate the selected instructions into model context, and verify that the configured skill is activated without changing unrelated conversations. Record the exact verification script, observed results, and limitations after testing.
