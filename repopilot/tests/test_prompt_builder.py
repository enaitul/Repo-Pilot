from repopilot.prompt_builder import PromptBuilder


def test_returns_system_and_user_prompt_pair():
    system_prompt, user_prompt = PromptBuilder.build("some context", "some question")
    assert isinstance(system_prompt, str)
    assert isinstance(user_prompt, str)


def test_user_prompt_contains_the_question():
    _, user_prompt = PromptBuilder.build("context here", "How does login work?")
    assert "How does login work?" in user_prompt


def test_user_prompt_contains_the_context():
    _, user_prompt = PromptBuilder.build("def login(): pass", "question")
    assert "def login(): pass" in user_prompt


def test_system_prompt_instructs_grounding():
    system_prompt, _ = PromptBuilder.build("context", "question")
    assert "context" in system_prompt.lower()
    assert "invent" in system_prompt.lower() or "not invent" in system_prompt.lower()


def test_system_prompt_instructs_admitting_insufficient_context():
    system_prompt, _ = PromptBuilder.build("context", "question")
    assert "does not contain enough information" in system_prompt.lower() or "say so" in system_prompt.lower()


def test_system_prompt_is_the_same_regardless_of_question():
    system_prompt_a, _ = PromptBuilder.build("context A", "question A")
    system_prompt_b, _ = PromptBuilder.build("context B", "question B")
    assert system_prompt_a == system_prompt_b
