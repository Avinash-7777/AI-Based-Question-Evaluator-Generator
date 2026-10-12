
from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional


# ============================================================
# CONFIGURATION
# ============================================================

OLLAMA_URL = os.getenv(
    "OLLAMA_URL",
    "http://127.0.0.1:11434/api/generate",
)

OLLAMA_MODEL = os.getenv(
    "OLLAMA_MODEL",
    "qwen3:4b-instruct",
)

OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "180"))

VALID_BLOOM_LEVELS = [
    "Remember",
    "Understand",
    "Apply",
    "Analyze",
    "Evaluate",
    "Create",
]

VALID_DIFFICULTIES = ["Easy", "Medium", "Hard"]


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_bloom(value: Any) -> str:
    if value is None:
        return "Understand"

    value = str(value).strip().lower()

    mapping = {
        "remember": "Remember",
        "remembering": "Remember",
        "recall": "Remember",
        "understand": "Understand",
        "understanding": "Understand",
        "apply": "Apply",
        "applying": "Apply",
        "application": "Apply",
        "analyze": "Analyze",
        "analyse": "Analyze",
        "analysis": "Analyze",
        "evaluating": "Evaluate",
        "evaluate": "Evaluate",
        "evaluation": "Evaluate",
        "create": "Create",
        "creating": "Create",
        "creation": "Create",
    }

    return mapping.get(value, "Understand")


def normalize_difficulty(value: Any) -> str:
    if value is None:
        return "Medium"

    value = str(value).strip().lower()

    mapping = {
        "easy": "Easy",
        "simple": "Easy",
        "low": "Easy",
        "medium": "Medium",
        "moderate": "Medium",
        "balanced": "Medium",
        "hard": "Hard",
        "difficult": "Hard",
        "high": "Hard",
    }

    return mapping.get(value, "Medium")


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_question(question: Any) -> str:
    """Clean model formatting without intentionally changing meaning."""

    if question is None:
        return ""

    question = str(question).strip()

    question = re.sub(
        r"<think>.*?</think>",
        "",
        question,
        flags=re.DOTALL | re.IGNORECASE,
    ).strip()

    question = re.sub(
        r"^```(?:json|text|python)?\s*|\s*```$",
        "",
        question,
        flags=re.IGNORECASE,
    ).strip()

    question = re.sub(
        r"^\s*(?:Q(?:uestion)?\s*)?\d+\s*[.):\-]\s*",
        "",
        question,
        flags=re.IGNORECASE,
    )

    question = re.sub(
        r"^\s*[-*•]\s*",
        "",
        question,
    )

    question = re.sub(r"\s+", " ", question).strip()

    question = re.sub(r"[?.!]+$", "", question).strip()

    if not question:
        return ""

    # Avoid returning a whole JSON object as a question.
    if question.startswith("{") or question.startswith("["):
        return ""

    return question + "?"


# ============================================================
# SOURCE-REFERENCE PHRASE FILTER
# ============================================================

FORBIDDEN_SOURCE_PHRASES = [
    "according to the text",
    "according to the given text",
    "according to the provided text",
    "according to the reference",
    "according to the reference material",
    "according to the provided reference",
    "according to the source",
    "according to the book",
    "according to the textbook",
    "as discussed in the text",
    "as mentioned in the text",
    "as stated in the text",
    "as described in the text",
    "as discussed in the reference",
    "as mentioned in the reference",
    "as stated in the reference",
    "as defined in the book",
    "as defined in the text",
    "as explained in the book",
    "as described in the book",
    "as presented in the book",
    "as stated in the book",
    "as given in the book",
    "as mentioned in the book",
    "as discussed in the book",
    "as defined in the provided material",
    "based on the text",
    "based on the given text",
    "based on the provided text",
    "based on the reference",
    "based on the reference material",
    "based on the provided reference",
    "from the given text",
    "from the provided text",
    "from the reference",
    "from the reference material",
    "from the provided reference",
    "in the given text",
    "in the provided text",
    "in the reference",
    "in the reference material",
    "in the provided reference",
    "in the textbook",
    "in the given passage",
    "from the passage",
    "according to the passage",
    "as discussed in the passage",
]


def contains_source_reference_phrase(question: str) -> bool:
    if not question:
        return True

    lowered = question.lower()

    return any(
        phrase in lowered
        for phrase in FORBIDDEN_SOURCE_PHRASES
    )


# ============================================================
# QUESTION QUALITY FILTER
# ============================================================


def question_is_reasonable(question: str) -> bool:
    if not isinstance(question, str) or not question.strip():
        return False

    question = re.sub(r"\s+", " ", question).strip()

    # Normalize repeated or misplaced punctuation.
    question = re.sub(r"\?+", "?", question)
    question = re.sub(r",\s*\?", "?", question)
    question = re.sub(r"\s+([,.;:!?])", r"\1", question)

    words = question.split()

    if len(words) < 5 or len(words) > 90:
        return False

    lowered = question.lower()

    # Reject answers accidentally included in the question.
    forbidden_answer_markers = [
        "answer:",
        "solution:",
        "correct answer:",
        "the answer is",
        "explanation:",
    ]

    if any(marker in lowered for marker in forbidden_answer_markers):
        return False

    # Reject phrases that refer to a source instead of asking
    # a standalone examination question.
    if contains_source_reference_phrase(question):
        return False

    # Reject questions that contain no meaningful question structure.
    if not re.search(r"[?]$", question):
        return False

    # Reject multiple consecutive punctuation marks.
    if re.search(r"[!?]{2,}", question):
        return False

    return True


def understand_question_is_valid(question: str) -> bool:
    if not question_is_reasonable(question):
        return False

    lowered = question.lower().strip()

    recall_starts = [
        "what is ",
        "what are ",
        "define ",
        "list ",
        "name ",
        "identify ",
        "which ",
        "who ",
        "when ",
        "where ",
    ]

    if any(lowered.startswith(prefix) for prefix in recall_starts):
        return False

    comparison_terms = [
        "compare",
        "contrast",
        "differentiate",
        "distinguish",
    ]

    if any(term in lowered for term in comparison_terms):
        return False

    # Look for the intended cognitive operation.
    preferred_terms = [
        "explain",
        "describe",
        "why",
        "how",
        "illustrate",
        "discuss",
        "interpret",
        "summarize",
        "summarise",
    ]

    return any(term in lowered for term in preferred_terms)


# ============================================================
# DUPLICATE DETECTION
# ============================================================

def normalize_duplicate_text(text: str) -> str:
    if not text:
        return ""

    text = text.lower()
    text = text.replace("–", "-").replace("—", "-")
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def remove_duplicate_questions(
    questions: List[str],
) -> List[str]:
    unique = []
    seen = set()

    for question in questions:
        normalized = normalize_duplicate_text(question)

        if not normalized or normalized in seen:
            continue

        seen.add(normalized)
        unique.append(question)

    return unique


def question_similarity(first: str, second: str) -> float:
    first_norm = normalize_duplicate_text(first)
    second_norm = normalize_duplicate_text(second)

    if not first_norm or not second_norm:
        return 0.0

    sequence_score = SequenceMatcher(
        None,
        first_norm,
        second_norm,
    ).ratio()

    first_words = set(first_norm.split())
    second_words = set(second_norm.split())

    union = first_words | second_words

    jaccard_score = (
        len(first_words & second_words) / len(union)
        if union
        else 0.0
    )

    return max(sequence_score, jaccard_score)


def is_duplicate_against_existing(
    candidate: str,
    existing: List[str],
    threshold: float = 0.82,
) -> bool:
    candidate_norm = normalize_duplicate_text(candidate)

    if not candidate_norm:
        return True

    for previous in existing:
        previous_norm = normalize_duplicate_text(previous)

        if candidate_norm == previous_norm:
            return True

        if question_similarity(candidate, previous) >= threshold:
            return True

    return False


# ============================================================
# NORMALIZE GENERATED QUESTIONS
# ============================================================

def normalize_generated_questions(
    raw_questions: Any,
) -> List[str]:
    if raw_questions is None:
        return []

    if isinstance(raw_questions, str):
        raw_questions = [raw_questions]

    if not isinstance(raw_questions, list):
        return []

    cleaned = []

    for item in raw_questions:
        if isinstance(item, dict):
            question = (
                item.get("question")
                or item.get("text")
                or item.get("query")
                or ""
            )
        else:
            question = item

        question = clean_question(question)

        if not question_is_reasonable(question):
            continue

        cleaned.append(question)

    return remove_duplicate_questions(cleaned)


# ============================================================
# JSON EXTRACTION
# ============================================================

def extract_json(text: str) -> Any:
    if not text:
        return None

    text = re.sub(
        r"<think>.*?</think>",
        "",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    ).strip()

    # Direct JSON.
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        pass

    # JSON in a Markdown block.
    match = re.search(
        r"```(?:json)?\s*(.*?)\s*```",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )

    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Find the first valid JSON object or array.
    decoder = json.JSONDecoder()

    for index, character in enumerate(text):
        if character not in "[{":
            continue

        try:
            value, _ = decoder.raw_decode(text[index:])
            return value
        except json.JSONDecodeError:
            continue

    return None


# ============================================================
# OLLAMA REQUEST
# ============================================================

def call_ollama(
    prompt: str,
    temperature: float = 0.25,
    top_p: float = 0.9,
    timeout: Optional[int] = None,
) -> str:
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "top_p": top_p,
        },
    }

    body = json.dumps(payload).encode("utf-8")

    request = urllib.request.Request(
        OLLAMA_URL,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    timeout_value = timeout if timeout is not None else OLLAMA_TIMEOUT

    try:
        with urllib.request.urlopen(
            request,
            timeout=timeout_value,
        ) as response:
            raw = response.read().decode("utf-8")

        parsed = json.loads(raw)

        return str(parsed.get("response", "")).strip()

    except urllib.error.HTTPError as exc:
        try:
            detail = exc.read().decode(
                "utf-8",
                errors="ignore",
            )
        except Exception:
            detail = ""

        raise RuntimeError(
            f"Ollama HTTP {exc.code}: {detail}"
        ) from exc

    except urllib.error.URLError as exc:
        raise RuntimeError(
            "Could not connect to Ollama. "
            f"Check that Ollama is running at {OLLAMA_URL}."
        ) from exc

    except Exception as exc:
        raise RuntimeError(
            f"Ollama request failed: {exc}"
        ) from exc


# ============================================================
# STRICT GROUNDING RULES
# ============================================================

GROUNDING_RULES = """
STRICT REFERENCE-GROUNDING RULES

The supplied reference material is the only authoritative
source for subject-specific knowledge.

Every concept, term, relationship, definition, process, method,
property, condition, example, formula, and factual premise
introduced in a question must be supported by the references.

Do not use general world knowledge or pretrained knowledge to
fill gaps. Do not invent facts, examples, data, technologies,
formulas, scenarios, assumptions, or terminology.

A question must be answerable using only the supplied material.

If the material does not support enough distinct questions,
return fewer questions. Never invent questions to fill a quota.

Write standalone university examination questions. Do not
mention the text, reference, passage, document, book, or source.

Do not provide answers, hints, or explanations.
"""


# ============================================================
# BLOOM INSTRUCTIONS
# ============================================================

BLOOM_INSTRUCTIONS = {
    "Remember": """
Ask the student to recall or identify information explicitly
present in the references. Suitable verbs include define, state,
list, identify, name, and recall.
""",
    "Understand": """
Ask the student to demonstrate understanding of a supported
concept or relationship. Suitable operations include explain,
describe, interpret, summarize, illustrate, why, and how.
Avoid a simple definition/list question or a comparison as the
primary task.
""",
    "Apply": """
Ask the student to use a method, rule, process, formula, or
concept explicitly supported by the references. Do not add
unsupported facts or scenarios.
""",
    "Analyze": """
Ask the student to examine supported relationships, components,
steps, causes, conditions, differences, or structure.
Keep the reasoning within the supplied material.
""",
    "Evaluate": """
Ask the student to assess an approach or justify a choice using
criteria, properties, limitations, advantages, or relationships
supported by the references. Do not invent criteria.
""",
    "Create": """
Ask the student to construct, design, formulate, or develop
something using only concepts and constraints supported by the
references. Do not invent external requirements or facts.
""",
}


# ============================================================
# QUESTION-TYPE INSTRUCTIONS
# ============================================================

QUESTION_TYPE_INSTRUCTIONS = {
    "short_concept": """
Create a concise short-mark question focused on one supported
concept.
""",
    "short_explanation": """
Ask for an explanation of one supported concept, process,
relationship, or reason.
""",
    "short_application": """
Ask the student to apply a supported concept, rule, method,
or process in a compact task.
""",
    "difference": """
Ask the student to analyze a meaningful difference or
relationship between supported concepts. Do not merely ask
for two definitions.
""",
    "comparison": """
Ask the student to compare supported concepts, methods,
processes, or properties using only reference information.
""",
    "application": """
Ask the student to apply a reference-supported method,
procedure, concept, or rule to a supported task.
""",
    "evaluation": """
Ask the student to evaluate or justify an approach using
reference-supported criteria and relationships.
""",
    "integrated_problem": """
Combine two or more related reference-supported concepts.
The combination must remain logically supported by the material.
""",
    "extended_problem": """
Create a deeper, multi-step task using several supported
concepts, processes, relationships, or conditions.
""",
    "mixed": """
Create a suitable examination question matching the target
Bloom level and marks.
""",
}


# ============================================================
# CONTEXT LIMIT
# ============================================================

def limit_context(
    context: str,
    max_chars: int = 30000,
) -> str:
    if not context:
        return ""

    context = str(context).strip()

    if len(context) <= max_chars:
        return context

    print(
        f"[CONTEXT] Truncating context from "
        f"{len(context)} to {max_chars} characters."
    )

    return context[:max_chars]


# ============================================================
# NORMAL GENERATION PROMPT
# ============================================================

def create_generation_prompt(
    topic: str,
    subtopic: str,
    context: str,
    number_of_questions: int,
    bloom: str,
    difficulty: str,
) -> str:
    bloom = normalize_bloom(bloom)
    difficulty = normalize_difficulty(difficulty)
    context = limit_context(context)

    return f"""
You are an academic examination question generator.

Generate up to {number_of_questions} distinct undergraduate
engineering examination questions.

SUBJECT OR TOPIC:
{topic}

SUBTOPIC:
{subtopic or "Not specified"}

TARGET BLOOM LEVEL:
{bloom}

TARGET DIFFICULTY:
{difficulty}

BLOOM REQUIREMENTS:
{BLOOM_INSTRUCTIONS[bloom]}

{GROUNDING_RULES}

REFERENCE MATERIAL
==================
{context}
==================

ADDITIONAL REQUIREMENTS:
1. Use only the supplied reference material.
2. Do not add outside knowledge or unsupported examples.
3. Each question must be clear and self-contained.
4. Avoid repeated questions and superficial rewordings.
5. Do not mention the reference or source in the question.
6. Match the requested Bloom level and difficulty.
7. Do not output answers or explanations.
8. Return valid JSON only.

Output format:
{{
  "questions": [
    {{"question": "Question text?"}}
  ]
}}

Return fewer questions if the material does not support the
requested number. Never invent a question to fill the quota.
""".strip()


# ============================================================
# NORMAL QUESTION GENERATION
# ============================================================

def generate_questions(
    topic: str,
    subtopic: str = "",
    context: str = "",
    number_of_questions: int = 3,
    bloom: Optional[str] = "Understand",
    difficulty: Optional[str] = "Medium",
) -> List[Dict[str, Any]]:
    if not isinstance(context, str) or not context.strip():
        print("[GENERATION] Missing reference context. Blocked.")
        return []

    try:
        number_of_questions = max(
            1,
            min(int(number_of_questions), 30),
        )
    except (TypeError, ValueError):
        number_of_questions = 3

    bloom = normalize_bloom(bloom)
    difficulty = normalize_difficulty(difficulty)

    prompt = create_generation_prompt(
        topic=topic,
        subtopic=subtopic,
        context=context,
        number_of_questions=number_of_questions,
        bloom=bloom,
        difficulty=difficulty,
    )

    try:
        raw = call_ollama(
            prompt,
            temperature=0.25,
            top_p=0.9,
        )

        print(f"[GENERATION] Raw response length: {len(raw)}")

        parsed = extract_json(raw)

        if isinstance(parsed, dict):
            raw_questions = parsed.get("questions", [])
        elif isinstance(parsed, list):
            raw_questions = parsed
        else:
            raw_questions = []

        questions = normalize_generated_questions(raw_questions)

        if bloom == "Understand":
            questions = [
                question
                for question in questions
                if understand_question_is_valid(question)
            ]

        result = []

        for question in questions[:number_of_questions]:
            result.append({
                "question": question,
                "target_bloom": bloom,
                "target_difficulty": difficulty,
            })

        print(
            f"[GENERATION] Generated "
            f"{len(result)}/{number_of_questions}"
        )

        return result

    except Exception as exc:
        print(f"[GENERATION] Error: {exc}")
        return []


# ============================================================
# PAPER QUESTION PROMPT
# ============================================================

# ============================================================
# PAPER QUESTION PROMPT
# ============================================================



def create_paper_question_prompt(
    context: str,
    number_of_questions: int,
    difficulty: str,
    previous_questions: Optional[List[str]] = None,
    selected_topics: Optional[List[str]] = None,
    bloom_targets: Optional[List[str]] = None,
    question_types: Optional[List[str]] = None,
    marks_targets: Optional[List[int]] = None,
) -> str:
    """Build a strict, reference-grounded examination question prompt."""

    previous_questions = previous_questions or []
    selected_topics = selected_topics or []
    bloom_targets = bloom_targets or []
    question_types = question_types or []
    marks_targets = marks_targets or []

    context = limit_context(context, max_chars=40000)
    difficulty = normalize_difficulty(difficulty)

    # This function is called per slot by the current paper generator.
    # Keep the slot requirements explicit and stable.
    bloom = normalize_bloom(
        bloom_targets[0] if bloom_targets else "Understand"
    )

    qtype = (
        str(question_types[0]).strip().lower()
        if question_types else "mixed"
    )

    marks = marks_targets[0] if marks_targets else 2

    type_instruction = QUESTION_TYPE_INSTRUCTIONS.get(
        qtype,
        QUESTION_TYPE_INSTRUCTIONS["mixed"],
    )

    bloom_instruction = BLOOM_INSTRUCTIONS[bloom]

    previous_text = "\n".join(
        f"- {question}" for question in previous_questions[-40:]
    ) or "None"

    topics_text = (
        ", ".join(selected_topics)
        if selected_topics
        else "Use relevant concepts explicitly supported by the reference material."
    )

    return f"""
You generate examination questions for undergraduate engineering students.

SUBJECT:
Use only the subject and concepts supported by the supplied reference material.
Do not select unrelated facts merely because they occur somewhere in a textbook.

TOPIC SCOPE:
{topics_text}

SLOT REQUIREMENTS:
Marks: {marks}
Required Bloom level: {bloom}
Difficulty: {difficulty}
Question type: {qtype}

Required cognitive operation:
{bloom_instruction}

Required question depth:
{type_instruction}

STRICT REFERENCE-GROUNDING POLICY:
{GROUNDING_RULES}

The supplied reference material is the sole source of subject knowledge.
Every technical term, factual premise, method, relationship, and condition
used in a question must be supported by that material.

Do not use pretrained knowledge to fill gaps.
Do not invent examples, scenarios, data, formulas, assumptions, or requirements.
Do not ask about a fact merely because it appears in the material if it is
unrelated to the requested subject or selected topics.
A question must be answerable using the supplied material alone.

NOVELTY:
Previously generated questions:
{previous_text}

Do not repeat or paraphrase these questions.
Do not test the same fact or learning objective with slightly different wording.
Prefer a different supported concept when one is available.
If no distinct, supported question is available, return fewer questions.

BLOOM ACCURACY:
The cognitive operation—not merely the opening verb—must match {bloom}.

Remember: recall a stated fact or concept.
Understand: explain a stated concept or relationship.
Apply: apply a referenced method or rule to a supported task.
Analyze: examine supported relationships, components, or differences.
Evaluate: make a judgment using explicit criteria or evidence in the references.
Create: construct or design using referenced concepts and constraints.

Do not label a recall question as Apply, Analyze, Evaluate, or Create.
Do not invent a scenario just to achieve a higher Bloom level.
If the requested cognitive operation cannot be supported, return fewer questions.

MARKS AND QUESTION TYPE:
The depth must be appropriate for {marks} marks.
{type_instruction}

QUESTION WRITING:
- Ask one clear question.
- Do not include the answer, hints, or explanation.
- Do not mention the source, book, document, passage, or reference material.
- Never write "based on the reference material",
  "according to the provided text", or similar source-related wording.
- Write a natural, standalone examination question.
- Do not invent a design, architecture, example, or requirement
  unless the supplied reference material supports it.
- Avoid vague phrases such as "discuss the significance" unless the requested
  task and supporting evidence are clear.
- Use natural, grammatically correct academic English.

REFERENCE MATERIAL:
===================
{context}
===================

Before returning a question, check that it is relevant, answerable from
the references, novel, and aligned with the required Bloom level.

Return valid JSON only, with no Markdown or commentary:
{{
    "questions": [
        {{"question": "Question text?"}}
    ]
}}

Return up to {number_of_questions} questions. Return fewer rather than
inventing unsupported content.
""".strip()

# ============================================================
# PAPER BATCH GENERATION
# ============================================================





def generate_paper_batch(
    subject: str,
    context: str,
    number_of_questions: int,
    difficulty: str = "balanced",
    previous_questions: Optional[List[str]] = None,
    selected_topics: Optional[List[str]] = None,
    bloom_targets: Optional[List[str]] = None,
    question_types: Optional[List[str]] = None,
    marks_targets: Optional[List[int]] = None,
) -> List[Dict[str, Any]]:
    """Generate unique examination questions grounded only in references."""

    if not isinstance(context, str) or not context.strip():
        print("[PAPER] Missing reference context. Blocked.")
        return []

    try:
        number_of_questions = max(1, min(int(number_of_questions), 30))
    except (TypeError, ValueError):
        number_of_questions = 1

    previous_questions = previous_questions or []
    selected_topics = selected_topics or []
    bloom_targets = bloom_targets or ["Understand"]
    question_types = question_types or ["mixed"]
    marks_targets = marks_targets or [2]

    excluded_questions = list(dict.fromkeys(
        str(question).strip()
        for question in previous_questions
        if str(question).strip()
    ))

    target_bloom = normalize_bloom(bloom_targets[0])
    target_type = question_types[0]
    target_marks = marks_targets[0]
    target_difficulty = normalize_difficulty(difficulty)

    topics_text = ", ".join(selected_topics) or subject
    bloom_text = ", ".join(
        normalize_bloom(level) for level in bloom_targets
    )
    type_text = ", ".join(str(value) for value in question_types)
    marks_text = ", ".join(str(value) for value in marks_targets)

    bloom_instructions = {
        "Remember": (
            "Prefer focused questions about facts, terms, or details "
            "explicitly stated in the references."
        ),
        "Understand": (
            "Prefer questions asking students to explain referenced "
            "concepts, purposes, processes, or relationships."
        ),
        "Apply": (
            "Prefer questions applying a referenced method or concept "
            "to a task that requires no unsupported assumptions."
        ),
        "Analyze": (
            "Prefer questions examining relationships, components, "
            "or processes explicitly supported by the references."
        ),
        "Evaluate": (
            "Prefer questions requiring a justified judgment using "
            "criteria or evidence supported by the references."
        ),
        "Create": (
            "Prefer questions requiring a design or construction from "
            "referenced concepts and constraints. Do not invent "
            "unsupported components or assumptions."
        ),
    }

    bloom_instruction = bloom_instructions.get(
        target_bloom,
        bloom_instructions["Understand"],
    )

    result: List[Dict[str, Any]] = []
    seen_in_batch: List[str] = []

    # Remove explicit references to source documents from question endings.
    source_phrase_patterns = [
        (
            r"(?i)\s*,?\s*(?:as described in|as stated in|"
            r"according to|based on)\s+"
            r"(?:the\s+)?(?:provided\s+)?"
            r"(?:reference material|references|source material|"
            r"provided text|text|passage)\s*$"
        ),
        (
            r"(?i)\s*,?\s*using only "
            r"(?:the\s+)?concepts and components explicitly "
            r"described in "
            r"(?:the\s+)?(?:provided\s+)?"
            r"(?:reference material|references|provided text|text)\s*$"
        ),
        (
            r"(?i)\s*,?\s*using only concepts explicitly "
            r"supported by (?:the\s+)?"
            r"(?:reference material|references|provided text|text)\s*$"
        ),
    ]

    # Bounded retries: do not generate indefinitely.
    max_attempts = 3

    try:
        for attempt in range(1, max_attempts + 1):
            remaining = number_of_questions - len(result)

            if remaining <= 0:
                break

            excluded_text = "\n".join(
                f"- {question}"
                for question in excluded_questions + seen_in_batch
            ) or "- No previous questions"

            prompt = create_paper_question_prompt(
                context=context,
                number_of_questions=remaining,
                difficulty=difficulty,
                previous_questions=excluded_questions + seen_in_batch,
                selected_topics=selected_topics,
                bloom_targets=bloom_targets,
                question_types=question_types,
                marks_targets=marks_targets,
            )

            prompt += f"""

ADDITIONAL STRICT GENERATION RULES

Subject: {subject}
Relevant topics: {topics_text}
Preferred Bloom level(s): {bloom_text}
Requested question type(s): {type_text}
Requested marks: {marks_text}
Questions still needed: {remaining}
Generation attempt: {attempt} of {max_attempts}

REFERENCE-ONLY POLICY:
- The supplied reference context is the sole source of facts.
- Every technical claim and concept must be supported by the context.
- Do not use outside knowledge or unsupported examples.
- Do not invent scenarios, facts, methods, constraints, or components.
- If the references cannot support another valid question, return fewer.

NOVELTY AND TOPIC DIVERSITY:
- Each question should test a distinct fact, concept, method,
  algorithm, architecture, or learning objective when supported.
- Read the previous questions and avoid their central concepts.
- Do not repeat a concept merely by changing wording or action verbs.
- Prefer under-tested concepts that are explicitly present in the context.
- Avoid repeatedly asking about historical names, professional societies,
  conference names, journal names, or dates unless they are important
  learning objectives supported by the references.
- Generate replacements for rejected duplicates when possible.

PREVIOUS QUESTIONS — DO NOT REPEAT:
{excluded_text}

BLOOM LEVEL POLICY:
- Prefer the requested Bloom level when practical.
- A valid question may naturally use a different Bloom level.
- Do not force an unrelated action verb into a question.
- Do not reject an otherwise valid question only because its
  cognitive level differs from the requested level.
- Never invent facts or concepts to satisfy a Bloom level.

TARGET PREFERENCE:
Preferred Bloom level: {target_bloom}
{bloom_instruction}

MARKS AND QUESTION COMPLEXITY:
- 2 marks: one focused fact, concept, or brief explanation.
- 5 marks: a developed explanation, comparison, or straightforward
  application when supported by the references.
- 10 marks: a substantial explanation, analysis, or application
  involving multiple aspects when supported by the references.
- Do not make a simple recall question artificially complex by
  adding unrelated clauses.
- Match the requested question type when practical.

QUESTION WORDING:
- Write natural, standalone examination questions.
- Do not mention books, documents, passages, or reference material.
- Do not write "according to the text" or "based on the references".
- Do not include answers, hints, or explanations.
- Avoid combining unrelated questions into one sentence.
- Use an appropriate action verb for the intended cognitive task.
- Return valid JSON only, without Markdown or extra text.

OUTPUT FORMAT:
{{"questions": [{{"question": "..."}}]}}

"""

            raw = call_ollama(
                prompt,
                temperature=0.30 if attempt == 1 else 0.40,
                top_p=0.9,
            )

            print(
                f"[PAPER DEBUG] Raw Qwen response, attempt {attempt}:"
            )
            print(raw[:3000])

            parsed = extract_json(raw)

            if isinstance(parsed, dict):
                raw_questions = parsed.get("questions", [])
            elif isinstance(parsed, list):
                raw_questions = parsed
            else:
                raw_questions = []

            if not isinstance(raw_questions, list):
                print("[PAPER] Invalid questions format.")
                continue

            accepted_this_attempt = 0

            for item in raw_questions:
                if isinstance(item, dict):
                    question_text = (
                        item.get("question")
                        or item.get("text")
                        or ""
                    )
                elif isinstance(item, str):
                    question_text = item
                else:
                    continue

                question = clean_question(question_text)

                if not question:
                    print("[PAPER] Rejected: empty question.")
                    continue

                # Remove only explicit source-related disclaimers.
                for pattern in source_phrase_patterns:
                    question = re.sub(pattern, "", question)

                # Normalize whitespace and punctuation safely.
                question = re.sub(r"\s+", " ", question).strip()
                question = re.sub(
                    r"\s+([,.;:!?])",
                    r"\1",
                    question,
                )
                question = re.sub(r",\s*\?", "?", question)
                question = re.sub(r"[ \t]+", " ", question)
                question = question.rstrip(" \t\r\n.,;:!")

                if question and not question.endswith("?"):
                    question += "?"

                if not question_is_reasonable(question):
                    print(
                        f"[PAPER] Rejected after cleanup: {question!r}"
                    )
                    continue

                # Preserve duplicate protection across attempts and slots.
                if is_duplicate_against_existing(
                    question,
                    excluded_questions + seen_in_batch,
                    threshold=0.78,
                ):
                    print(
                        "[PAPER] Rejected: duplicate or near-duplicate."
                    )
                    continue

                result.append({
                    "question": question,
                    "target_bloom": target_bloom,
                    "target_difficulty": target_difficulty,
                    "question_type": target_type,
                    "marks": target_marks,
                })

                seen_in_batch.append(question)
                accepted_this_attempt += 1

                if len(result) >= number_of_questions:
                    break

            print(
                f"[PAPER] Attempt {attempt}: "
                f"accepted {accepted_this_attempt}; "
                f"total {len(result)}/{number_of_questions}"
            )

        print(
            f"[PAPER] Batch generated "
            f"{len(result)}/{number_of_questions}"
        )

        return result

    except Exception as exc:
        print(f"[PAPER] Error during batch generation: {exc}")
        return result

# ============================================================
# QWEN QUESTION CLASSIFICATION
# ============================================================

def classify_question_with_llm(
    question: str,
) -> Dict[str, Any]:
    question = str(question or "").strip()

    if not question:
        return {
            "success": False,
            "bloom": None,
            "difficulty": None,
            "confidence": 0.0,
            "error": "Question is empty.",
        }

    prompt = f"""
You are an academic examination question classifier.

Classify the following university examination question.

QUESTION:
{question}

Choose exactly one Bloom level:
Remember, Understand, Apply, Analyze, Evaluate, Create.

Choose exactly one difficulty:
Easy, Medium, Hard.

Classify the actual cognitive operation required, not merely
the presence of an action word.

Return JSON only:
{{
  "bloom": "Understand",
  "difficulty": "Medium",
  "confidence": 0.80
}}

Confidence must be a number between 0.0 and 1.0.
""".strip()

    try:
        raw = call_ollama(
            prompt,
            temperature=0.10,
            top_p=0.8,
        )

        parsed = extract_json(raw)

        if not isinstance(parsed, dict):
            return {
                "success": False,
                "bloom": None,
                "difficulty": None,
                "confidence": 0.0,
                "error": "Invalid classifier JSON.",
            }

        try:
            confidence = float(parsed.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0

        confidence = max(0.0, min(1.0, confidence))

        return {
            "success": True,
            "bloom": normalize_bloom(parsed.get("bloom")),
            "difficulty": normalize_difficulty(
                parsed.get("difficulty")
            ),
            "confidence": confidence,
        }

    except Exception as exc:
        print(f"[QWEN CLASSIFY] Error: {exc}")

        return {
            "success": False,
            "bloom": None,
            "difficulty": None,
            "confidence": 0.0,
            "error": str(exc),
        }


# ============================================================
# QUESTION VALIDATION WITH QWEN
# ============================================================


def validate_question_with_llm(
    question: str,
    context: str,
) -> Dict[str, Any]:
    """
    Verify that a generated question is answerable using only
    the supplied reference material. Fail closed on errors.
    """

    question = str(question or "").strip()
    context = str(context or "").strip()

    if not question:
        return {
            "valid": False,
            "grounded": False,
            "confidence": 0.0,
            "reason": "The question is empty.",
        }

    if not context:
        return {
            "valid": False,
            "grounded": False,
            "confidence": 0.0,
            "reason": "No reference material was supplied.",
        }

    prompt = f"""
You are a strict academic question-grounding validator.

Your task is to check ONE QUESTION against the supplied REFERENCE MATERIAL.

The question to verify is explicitly provided below. Do not claim that
the question is missing. Read the complete QUESTION TO VERIFY section.

QUESTION TO VERIFY:
<QUESTION_TO_VERIFY>
{question}
</QUESTION_TO_VERIFY>

REFERENCE MATERIAL:
<REFERENCE_MATERIAL>
{context}
</REFERENCE_MATERIAL>

VALIDATION RULES:
1. Determine whether the reference material directly supports the
   concepts and facts required to answer the question.
2. Set grounded=true only when the question is supported by the
   supplied reference material.
3. Set grounded=false if the question requires outside knowledge,
   introduces unsupported concepts, or cannot be answered from
   the references.
4. Do not use your own general knowledge to fill gaps in the references.
5. Set valid=true only if the question is meaningful, academically
   appropriate, and grounded in the supplied reference material.
6. A question does not need to repeat the exact wording of the source.
   Paraphrasing is acceptable when the meaning remains supported.
7. If the references are insufficient or ambiguous, reject the question.
8. The question itself is the item being checked, not a sentence that
   must already appear verbatim in the reference material.

Return ONLY one JSON object in this exact format:
{{
  "valid": true,
  "grounded": true,
  "confidence": 0.95,
  "reason": "Brief explanation based only on the supplied references."
}}

Use boolean values for valid and grounded.
Confidence must be a number from 0.0 to 1.0.
If validation fails, set valid=false and grounded=false.
Do not return Markdown or additional text.
"""

    try:
        response = call_ollama(
            prompt,
            temperature=0.0,
            top_p=0.1,
        )

        parsed = extract_json(response)

        if not isinstance(parsed, dict):
            return {
                "valid": False,
                "grounded": False,
                "confidence": 0.0,
                "reason": "Validator returned invalid JSON.",
            }

        valid_value = parsed.get("valid")
        grounded_value = parsed.get("grounded")

        # Fail closed: only explicit JSON booleans are accepted.
        if valid_value is not True or grounded_value is not True:
            return {
                "valid": False,
                "grounded": False,
                "confidence": 0.0,
                "reason": str(
                    parsed.get(
                        "reason",
                        "The question could not be verified against the references.",
                    )
                ),
            }

        try:
            confidence = float(parsed.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0

        confidence = max(0.0, min(1.0, confidence))

        return {
            "valid": True,
            "grounded": True,
            "confidence": confidence,
            "reason": str(
                parsed.get(
                    "reason",
                    "Supported by the supplied reference material.",
                )
            ),
        }

    except Exception as exc:
        print(f"[GROUNDING] Validator error: {exc}")

        return {
            "valid": False,
            "grounded": False,
            "confidence": 0.0,
            "reason": "Grounding validation failed because of an internal error.",
        }

    
# ============================================================
# OPTIONAL QUESTION REFINEMENT
# ============================================================

def refine_question_with_qwen(
    question: str,
    context: str,
    bloom: Optional[str] = None,
    difficulty: Optional[str] = None,
) -> str:
    """
    Refine wording conservatively.

    The original question is returned if the refinement fails.
    """

    question = str(question or "").strip()
    context = str(context or "").strip()

    if not question:
        return ""

    if not context:
        return question

    bloom = normalize_bloom(bloom)
    difficulty = normalize_difficulty(difficulty)

    prompt = f"""
Rewrite the examination question so it is clearer and more
natural, without changing the concept or cognitive task.

STRICT RULES:
- Use only concepts supported by the reference material.
- Do not add facts, examples, assumptions, or requirements.
- Preserve the original intended task.
- Do not mention the text, reference, book, source, or passage.
- Do not provide an answer or explanation.
- Return only the revised question.

TARGET BLOOM:
{bloom}

TARGET DIFFICULTY:
{difficulty}

ORIGINAL QUESTION:
{question}

REFERENCE MATERIAL:
{limit_context(context, 20000)}
""".strip()

    try:
        raw = call_ollama(
            prompt,
            temperature=0.10,
            top_p=0.8,
        )

        parsed = extract_json(raw)

        if isinstance(parsed, dict):
            raw_question = parsed.get("question", "")
        else:
            raw_question = raw

        refined = clean_question(raw_question)

        if not question_is_reasonable(refined):
            return question

        if contains_source_reference_phrase(refined):
            return question

        # Reject refinements that introduce terms absent from
        # the original question and reference context together.
        reference_words = set(
            normalize_duplicate_text(context).split()
        )

        original_words = set(
            normalize_duplicate_text(question).split()
        )

        refined_words = set(
            normalize_duplicate_text(refined).split()
        )

        permitted_words = reference_words | original_words

        meaningful_new_words = {
            word
            for word in refined_words - permitted_words
            if len(word) > 3
        }

        if meaningful_new_words:
            print(
                "[REFINE] Rejected refinement with unsupported "
                f"vocabulary: {sorted(meaningful_new_words)}"
            )
            return question

        return refined

    except Exception as exc:
        print(f"[REFINE] Error: {exc}")
        return question


# ============================================================
# COMPATIBILITY ALIAS
# ============================================================

def generate_question(
    topic: str,
    subtopic: str = "",
    context: str = "",
    bloom: str = "Understand",
    difficulty: str = "Medium",
) -> Optional[Dict[str, Any]]:
    questions = generate_questions(
        topic=topic,
        subtopic=subtopic,
        context=context,
        number_of_questions=1,
        bloom=bloom,
        difficulty=difficulty,
    )

    if not questions:
        return None

    return questions[0]