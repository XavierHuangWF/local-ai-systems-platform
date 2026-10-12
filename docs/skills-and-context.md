# Skills and Dynamic Context Management

## Objective and Implementation Status

Implement task-specific instructions as versioned Markdown skill definitions and load them into the Agent Harness without hard-coding them in Python.

**Implemented and verified:**

- A `research_summary` skill definition and reusable `SkillLoader`.
- Explicit skill activation and deactivation in `AgentHarness`.
- Insertion of active skill instructions into the system message submitted to the model client.
- Conversation reset when the active skill changes, while ordinary reset retains the active skill.
- Three Skill Loader verification checks and six Skill Activation verification checks.
- Twelve passing automated unit tests covering Agent Harness behavior and skill activation.
- One successful real-model skill integration verification using Qwen3.

**Real-model verification:** The `research_summary` skill was successfully tested with Qwen3 using one synthetic research experiment. All five required output sections were generated. Reliability across multiple examples has not yet been evaluated.

## Architecture

```text
skills/research_summary/SKILL.md
             |
             v
     SkillLoader.load()
             |
             v
 Skill(name, description, instructions)
             |
             v
 AgentHarness.activate_skill()
             |
             v
 General system instructions + active skill instructions
             |
             v
 Conversation history + current user request
             |
             v
 Token-budget validation
             |
             v
 VLLMClient -> vLLM -> Qwen3
```

Skill loading is a file-reading operation. A skill does not execute code, grant tool access, or independently invoke the model. The harness includes loaded instructions in the prompt when it calls the model client.

## Implementation

### Skill definition

**File:** `skills/research_summary/SKILL.md`

The file has `---`-delimited metadata containing its `name` and `description`, followed by Markdown instructions. The research-summary instructions define the expected sections, evidence requirements, quantitative-result accuracy, and explicit reporting of limitations.

### Skill loader

**File:** `src/skills/skill_loader.py`

`SkillLoader.load(skill_name)`:

- Resolves the configured skills directory relative to the project root.
- Validates the skill name against an allowed pattern.
- Resolves the requested `SKILL.md` and checks that its resolved path is inside the skills directory.
- Reads `---`-delimited metadata, requiring the declared name to match the directory.
- Rejects invalid names, missing files, and incomplete definitions.
- Returns a `Skill` data object with `name`, `description`, and `instructions`.

The metadata parser supports simple `key: value` fields, **not** the full YAML (YAML Ain't Markup Language) specification.

### Agent Harness integration

**File:** `src/agent/harness.py`

- `skill_loader`: Holds the reusable loader.
- `active_skill`: Holds the selected skill or `None`.
- `_system_content()`: Combines the general system instructions and active skill instructions.
- `activate_skill(name)`: Loads and validates the skill before changing state; then begins a new conversation.
- `deactivate_skill()`: Removes the active skill and starts a general-purpose conversation.
- `reset()`: Clears conversation history but retains the currently selected skill.

Activation is **explicit**, using `activate_skill("research_summary")`. Automatic skill selection and Command-Line Interface (CLI) skill commands have not been implemented.

## Verification: Skill Loader

**Script:** `scripts/verify_skill_loader.py`

**Command**, run from the repository root with the Python environment active:

```bash
python -m scripts.verify_skill_loader
```

A running vLLM server is not required.

### Observed results

```text
Test 1: Load existing skill
Skill name: research_summary
Description: Summarize technical research with evidence and limitations.
Instruction length: 1050 characters
PASS: Research summary skill loaded.

Test 2: Invalid skill name
Expected error: Invalid skill name.
PASS: Invalid skill name rejected.

Test 3: Nonexistent skill
Expected error: Skill not found: nonexistent_skill
PASS: Missing skill handled correctly.

PASS: All skill loader checks passed.
```

These checks establish local parsing, path/name validation, and missing-file behavior. They do not establish that skill instructions are safe or that the model follows them.

## Verification: Skill Activation

**Script:** `scripts/verify_skill_activation.py`

**Command**, run from the repository root with the Python environment active:

```bash
python -m scripts.verify_skill_activation
```

This script uses the real `AgentHarness` and `SkillLoader`, with simulated `VLLMClient` and `TokenCounter` components. A running model server or Graphics Processing Unit (GPU) inference is not required.

### Observed results

```text
Test 1: Initial State
PASS: No skill is active initially.

Test 2: Skill Activation
Active skill: research_summary
PASS: Skill instructions loaded.

Test 3: Model Request Context
PASS: Active skill included in model request.

Test 4: Conversation Reset
PASS: Reset preserves the active skill.

Test 5: Skill Deactivation
PASS: Skill deactivated and context restored.

Test 6: Failed Activation
PASS: Failed activation preserved existing state.

PASS: All six skill activation checks passed.
```

These tests verify how the harness prepares its messages and preserves state. They do not test Qwen3's behavior under the skill instructions.

## Debugging Record

### Initial failure

```text
ValueError: Missing skill metadata.
```

The first `SKILL.md` contained an unexpected leading newline. Inspection with Python's `repr()` function showed that the on-disk file started with `\n---` rather than `---`.

### Root cause and resolution

The parser requires the opening `---` delimiter on the first line. Removing the initial blank line corrected the file format. The three Skill Loader verification checks then passed.

## Engineering Decisions

### Separate instructions from application logic

Skill definitions are Markdown resources. They can be reviewed, versioned, and changed without modifying the Agent Harness implementation.

### Validate skill identifiers and file locations

Name validation and resolved-path checks reduce unintended file access. They do not make arbitrary skill content trustworthy or protect against prompt injection embedded inside an allowed skill file.

### Explicit activation and context isolation

Skill activation is deliberate rather than model-selected. Changing skills resets the prior conversation to avoid automatically carrying unrelated messages between tasks.

### Preserve state on failed activation

The requested skill is loaded and validated before `active_skill` and conversation history change. A missing skill leaves the previous state intact.

### Keep deterministic tests independent of inference

The Skill Loader and skill activation behavior can be checked without loading a language model. This isolates parsing and application-state failures from inference behavior.

## Current Limitations and Next Work

- Only `research_summary` is implemented; one skill can be active at a time.
- Skill selection is available through Python methods, not yet through an interactive CLI command.
- Skill activation passed deterministic unit tests and one real Qwen3 skill-conditioned integration verification; broader reliability remains unmeasured.
- The metadata reader supports only simple front matter rather than full YAML.
- Skill files inside the project are treated as trusted resources; malicious instruction content is not inspected.
- Skills do not execute external tools or grant new permissions.
- Concurrent activation and persistent skill/session state are not supported.

## Automated Regression Testing

### Objective

Verify that skill activation and deactivation work correctly
without breaking existing Agent Harness functionality.

### Implementation

- `tests/test_harness.py`: 7 Agent Harness unit tests.
- `tests/test_skill_activation.py`: 5 skill activation unit tests.

The tests use Python's unittest framework.

The real AgentHarness and SkillLoader are tested with simulated
model client and token-counter components.

### Reproduction

Run from the project root:

```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

### Actual Results

```text
Agent Harness tests: 7 passed
Skill Activation tests: 5 passed

Ran 12 tests in 0.120s

OK
```

### Verified Behaviors

- Conversation history preservation
- Empty-input rejection
- Conversation-history limits
- Model failure handling
- Oversized-input rejection
- Conversation reset
- Token-budget enforcement
- Initial skill state
- Skill instruction inclusion in model requests
- Skill preservation during reset
- Skill deactivation
- State preservation after failed skill activation

### Engineering Findings

All 12 automated unit tests passed.

The tests verify deterministic application behavior independently
of the local language model.

### Current Limitations

These unit tests establish application behavior with simulated model responses.
A separate real-model verification confirmed the required output structure
on one synthetic research example. Broader response quality and consistency
remain unmeasured.


## Real-Model Skill Integration Verification

### Objective

Verify that the local Qwen3 model can receive and follow
task-specific instructions loaded through the Agent Harness.

### Components

- `src/agent/harness.py`: Skill activation and conversation management.
- `src/skills/skill_loader.py`: Skill loading and validation.
- `skills/research_summary/SKILL.md`: Research-summary instructions.
- `scripts/verify_skill_integration.py`: Reproducible integration verification.
- vLLM: Local model inference server.
- Qwen3-8B-AWQ: Language model.

### Execution

Start the local inference server:

```bash
bash scripts/serve_vllm.sh
```

In another Ubuntu terminal, run from the project root
with the Python environment activated:

```bash
python -m scripts.verify_skill_integration
```

### Experimental Input

A fictional classification experiment containing:

- Objective: Improve classification using synthetic training data.
- Evaluation: 200 synthetic samples.
- Model A: Macro-averaged F1 score of 0.72.
- Model B: Macro-averaged F1 score of 0.80.
- Limitations: Synthetic evaluation data, one evaluation split,
  and no external validation.

### Actual Results

```text
PASS: Research summary skill activated.
PASS: Real model returned a response.
PASS: Skill instructions preserved in model context.
PASS: Required research summary sections present.

PASS: Real-model skill integration verification completed.
```

The generated response contained all five required headings:

1. Research Objective
2. Methodology
3. Key Findings
4. Limitations
5. Practical Implications

The response preserved the reported F1 scores of 0.72 and 0.80
and acknowledged the absence of external validation.

### Engineering Findings

The Agent Harness successfully loaded the research-summary skill
and included its instructions in the system message.

The real Qwen3 model generated a structured response matching
the required section headings.

The verification script automatically checked response presence,
skill-context retention, and required output headings.

The reported numerical values and study limitations were
inspected manually.

The response also contained empty `<think></think>` markers,
indicating that model-output formatting remains an improvement area.

### Limitations

- Only one synthetic research example was evaluated.
- The verification checks required headings but does not
  automatically evaluate factual accuracy.
- No comparison against a response without the skill was performed.
- Repeated-trial consistency has not been measured.
- Skill reliability across different document types remains unverified.
