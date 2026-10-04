import json
import os
import re
from typing import Any, Dict, List, Optional

import requests
from dotenv import load_dotenv


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()


# ============================================================
# OLLAMA CONFIG
# ============================================================

OLLAMA_URL = os.getenv(
    "OLLAMA_URL",
    "http://127.0.0.1:11434/api/generate"
)

OLLAMA_MODEL = os.getenv(
    "OLLAMA_MODEL",
    "qwen3:4b-instruct"
)

OLLAMA_TIMEOUT = int(
    os.getenv("OLLAMA_TIMEOUT", "180")
)


# ============================================================
# NORMALIZATION HELPERS
# ============================================================

def normalize_bloom(value: Optional[str]) -> str:
    if not value:
        return "Understand"

    value = str(value).strip().lower()

    mapping = {
        "remember": "Remember",
        "understand": "Understand",
        "apply": "Apply",
        "analyze": "Analyze",
        "analyse": "Analyze",
        "evaluate": "Evaluate",
        "create": "Create",
    }

    return mapping.get(value, "Understand")


def normalize_difficulty(value: Optional[str]) -> str:
    if not value:
        return "Medium"

    value = str(value).strip().lower()

    mapping = {
        "easy": "Easy",
        "medium": "Medium",
        "hard": "Hard",
    }

    return mapping.get(value, "Medium")


# ============================================================
# QUESTION CLEANING
# ============================================================

def clean_question(question: Any) -> str:
    if question is None:
        return ""

    question = str(question).strip()

    question = re.sub(
        r"^(?:Q\d+[\s:.)-]*|Question\s*\d+[\s:.)-]*)",
        "",
        question,
        flags=re.IGNORECASE,
    )

    question = question.strip().strip('"').strip("'").strip()

    question = re.sub(r"\s+", " ", question)

    question = re.sub(r"\s+\?", "?", question)

    if question and not question.endswith("?"):
        question += "?"

    return question.strip()


# ============================================================
# JSON EXTRACTION
# ============================================================

def extract_json(text: str) -> Any:
    if not text:
        return None

    text = text.strip()

    # Remove markdown code fences
    text = re.sub(
        r"```(?:json)?",
        "",
        text,
        flags=re.IGNORECASE,
    )

    text = text.replace("```", "").strip()

    # Direct JSON
    try:
        return json.loads(text)
    except Exception:
        pass

    # Find JSON array
    array_match = re.search(
        r"\[[\s\S]*\]",
        text,
    )

    if array_match:
        try:
            return json.loads(array_match.group(0))
        except Exception:
            pass

    # Find JSON object
    object_match = re.search(
        r"\{[\s\S]*\}",
        text,
    )

    if object_match:
        try:
            return json.loads(object_match.group(0))
        except Exception:
            pass

    return None


# ============================================================
# BLOOM-SPECIFIC INSTRUCTIONS
# ============================================================

def get_bloom_instruction(bloom: str) -> str:
    bloom = normalize_bloom(bloom)

    if bloom == "Remember":
        return """
BLOOM LEVEL: REMEMBER

The question must test direct recall of factual knowledge.

Good patterns:
- What is ...?
- Define ...
- Name ...
- List ...
- Identify ...

The answer should mainly require remembering a fact, term,
definition, component, property, or basic fact.

DO NOT require explanation, calculation, multi-step reasoning,
comparison, evaluation, or design.
"""

    if bloom == "Understand":
        return """
BLOOM LEVEL: UNDERSTAND

This is extremely important.

The question must test whether the student UNDERSTANDS a concept,
process, mechanism, relationship, or purpose.

The student should have to explain or describe something in their
own words, rather than simply recall a definition or fact.

PRIORITIZE THESE QUESTION PATTERNS:

1. Explain how ...
2. Describe how ...
3. Explain why ...
4. Describe the process by which ...
5. Explain the relationship between ...
6. Explain how one concept affects another.

PREFERRED EXAMPLES:

- Explain how information gain helps a decision tree select a
  splitting attribute.
- Describe how a decision tree determines the class of an input
  as it moves from the root toward a leaf.
- Explain how a regression tree produces a continuous prediction.
- Describe how entropy changes when the examples in a node become
  more homogeneous.
- Explain why pruning can improve the generalization of a
  decision tree.

IMPORTANT:

Prefer ONE clear concept and ONE clear explanatory relationship.

Avoid making the question unnecessarily comparative.

DO NOT prefer these patterns:
- Contrast X with Y
- Distinguish X from Y
- Compare X and Y
- Compare and contrast X and Y

These patterns can drift toward ANALYZE.

Also reject simple recall questions such as:
- What is X?
- What are the components of X?
- Define X.
- Name X.
- List X.
- Identify X.
- Which algorithm is used?
- Which attribute is selected?
- What does X measure?

Also avoid questions whose answer is just one memorized fact.

The ideal Understand question asks the student to explain
HOW, WHY, or HOW A PROCESS WORKS.

The question should still be answerable directly from the supplied
reference context.
"""

    if bloom == "Apply":
        return """
BLOOM LEVEL: APPLY

The question must require applying a known concept, rule,
algorithm, or procedure to a specific situation.

Prefer:
- Calculate ...
- Determine ...
- Apply ...
- Given a scenario, ...
- Trace ...
- Use ...

The student should perform or trace something rather than merely
define or explain it.
"""

    if bloom == "Analyze":
        return """
BLOOM LEVEL: ANALYZE

The question must require breaking information into parts,
examining relationships, comparing alternatives, identifying
patterns, or determining how components interact.

Prefer:
- Compare ...
- Analyze ...
- Examine ...
- Distinguish ...
- Contrast ...
- Determine why two approaches behave differently ...

The task should require reasoning beyond simple explanation.
"""

    if bloom == "Evaluate":
        return """
BLOOM LEVEL: EVALUATE

The question must require making a judgment using criteria,
evidence, trade-offs, or justification.

Prefer:
- Evaluate ...
- Justify ...
- Assess ...
- Critique ...
- Which approach is better and why?
- Defend ...

The student must make or support a judgment rather than simply
explain or apply a concept.
"""

    if bloom == "Create":
        return """
BLOOM LEVEL: CREATE

The question must require producing, designing, constructing,
formulating, or proposing something new.

Prefer:
- Design ...
- Construct ...
- Develop ...
- Propose ...
- Formulate ...
- Create ...

The student must generate an artifact, method, solution,
architecture, or approach.
"""

    return """
Use the requested Bloom level exactly.
"""


# ============================================================
# DIFFICULTY INSTRUCTIONS
# ============================================================

def get_difficulty_instruction(difficulty: str) -> str:
    difficulty = normalize_difficulty(difficulty)

    if difficulty == "Easy":
        return """
DIFFICULTY: EASY

Use a straightforward question based directly on the reference
material.

Requirements:
- One main concept.
- Clear wording.
- No unnecessary multi-step reasoning.
- No hidden tricks.
- No complex scenario.
- Answer should be obtainable directly from the context.
"""

    if difficulty == "Medium":
        return """
DIFFICULTY: MEDIUM

Require moderate reasoning or connection of related ideas from
the reference material.

Avoid excessive complexity.
"""

    if difficulty == "Hard":
        return """
DIFFICULTY: HARD

Require deeper reasoning, multiple connected concepts,
non-trivial application, or a more demanding scenario.

Do not make the question ambiguous.
"""

    return ""


# ============================================================
# QUESTION QUALITY CHECK
# ============================================================

def question_is_reasonable(question: str) -> bool:
    if not question:
        return False

    question_lower = question.lower().strip()

    # Too short
    if len(question_lower.split()) < 7:
        return False

    # Too long
    if len(question_lower.split()) > 60:
        return False

    # Remove obvious answer-like output
    forbidden_patterns = [
        "answer:",
        "solution:",
        "correct answer:",
    ]

    if any(pattern in question_lower for pattern in forbidden_patterns):
        return False

    return True


def understand_question_is_valid(question: str) -> bool:
    """
    Additional generation-time filter for Understand questions.

    We deliberately prefer canonical Understand wording because
    QDiff was trained on questions where Understand is commonly
    expressed using explain/describe-style formulations.
    """

    q = question.lower().strip()

    # Strong recall indicators
    recall_patterns = [
        r"^what is\b",
        r"^what are\b",
        r"^define\b",
        r"^name\b",
        r"^list\b",
        r"^identify\b",
        r"^which\b",
        r"^what does\b",
        r"^what do\b",
    ]

    for pattern in recall_patterns:
        if re.search(pattern, q):
            return False

    # Comparative wording can drift into Analyze.
    analyze_like_patterns = [
        r"^compare\b",
        r"^contrast\b",
        r"^distinguish\b",
        r"^compare and contrast\b",
    ]

    for pattern in analyze_like_patterns:
        if re.search(pattern, q):
            return False

    # Prefer canonical explanatory wording.
    preferred_patterns = [
        r"^explain\b",
        r"^describe\b",
        r"^why\b",
        r"^how does\b",
        r"^how do\b",
        r"^how is\b",
        r"^how are\b",
        r"^how can\b",
        r"^how does\b",
    ]

    if any(re.search(pattern, q) for pattern in preferred_patterns):
        return True

    return False


# ============================================================
# PROMPT CREATION
# ============================================================

def create_prompt(
    topic: str,
    subtopic: str,
    context: str,
    bloom: str,
    difficulty: str,
    number_of_questions: int,
) -> str:

    bloom = normalize_bloom(bloom)
    difficulty = normalize_difficulty(difficulty)

    bloom_instruction = get_bloom_instruction(bloom)
    difficulty_instruction = get_difficulty_instruction(difficulty)

    subtopic_text = subtopic.strip() if subtopic else "None specified"

    return f"""
You are an expert engineering question generator.

Generate exactly {number_of_questions} high-quality questions.

TOPIC:
{topic}

SUBTOPIC:
{subtopic_text}

REFERENCE CONTEXT:
{context}

============================================================
BLOOM REQUIREMENT
============================================================

{bloom_instruction}

============================================================
DIFFICULTY REQUIREMENT
============================================================

{difficulty_instruction}

============================================================
STRICT CONTENT RULES
============================================================

1. Use ONLY information supported by the reference context.
2. Do not invent facts.
3. Do not require information outside the context.
4. Do not include answers.
5. Do not include explanations after the questions.
6. Do not include Bloom labels.
7. Do not include difficulty labels.
8. Do not number questions inside the JSON values.
9. Every question must end with a question mark.
10. Avoid ambiguous wording.
11. Avoid duplicate questions.
12. Questions should be appropriate for engineering students.

============================================================
SPECIAL RULE FOR UNDERSTAND
============================================================

If Bloom = Understand:

The questions MUST primarily test explanation or conceptual
understanding.

Prefer:

"Explain how ..."
"Describe how ..."
"Explain why ..."
"Describe the process by which ..."

Avoid:

"What is ..."
"What are ..."
"Define ..."
"Name ..."
"List ..."
"Identify ..."
"Which ..."
"What does ..."
"Compare ..."
"Contrast ..."
"Distinguish ..."

For example:

BAD:
What does information gain measure?

GOOD:
Explain how information gain helps a decision tree select a
splitting attribute.

BAD:
Which attribute is selected using information gain?

GOOD:
Describe how information gain helps a decision tree choose
between candidate splitting attributes.

BAD:
Distinguish between a classification tree and a regression tree.

GOOD:
Explain how the outputs of a classification tree differ from
those of a regression tree.

The GOOD examples require conceptual explanation while remaining
simple enough for the Understand level.

============================================================
OUTPUT FORMAT
============================================================

Return ONLY valid JSON.

Use exactly this format:

{{
  "questions": [
    "Question 1?",
    "Question 2?",
    "Question 3?"
  ]
}}

No markdown.
No code fences.
No commentary.
"""


# ============================================================
# NORMALIZE GENERATED QUESTIONS
# ============================================================

def normalize_generated_questions(data: Any) -> List[str]:

    questions: List[str] = []

    if isinstance(data, dict):
        raw_questions = data.get("questions", [])

        if isinstance(raw_questions, list):
            questions.extend(raw_questions)

        elif isinstance(raw_questions, str):
            questions.append(raw_questions)

    elif isinstance(data, list):
        questions.extend(data)

    elif isinstance(data, str):
        lines = data.splitlines()

        for line in lines:
            line = line.strip()

            if not line:
                continue

            line = re.sub(
                r"^(?:\d+[\.\)]|[-*])\s*",
                "",
                line,
            )

            questions.append(line)

    cleaned: List[str] = []

    for question in questions:
        question = clean_question(question)

        if not question:
            continue

        if not question_is_reasonable(question):
            continue

        if question not in cleaned:
            cleaned.append(question)

    return cleaned


# ============================================================
# OLLAMA CALL
# ============================================================

def call_ollama(prompt: str) -> str:

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.25,
            "top_p": 0.9,
        },
    }

    response = requests.post(
        OLLAMA_URL,
        json=payload,
        timeout=OLLAMA_TIMEOUT,
    )

    response.raise_for_status()

    data = response.json()

    return str(data.get("response", "")).strip()


# ============================================================
# GENERATE QUESTIONS
# ============================================================

def generate_questions(
    topic: str,
    subtopic: str = "",
    context: str = "",
    number_of_questions: int = 3,
    bloom: str = "Understand",
    difficulty: str = "Medium",
    requested_bloom: Optional[str] = None,
    requested_difficulty: Optional[str] = None,
    **kwargs,
) -> List[str]:

    # Explicit requested values take priority.
    if requested_bloom:
        bloom = requested_bloom

    if requested_difficulty:
        difficulty = requested_difficulty

    bloom = normalize_bloom(bloom)
    difficulty = normalize_difficulty(difficulty)

    # Ask for a few extra candidates so that filtering does not
    # leave us with too few questions.
    candidate_count = max(
        number_of_questions + 2,
        number_of_questions,
    )

    candidate_count = min(candidate_count, 8)

    prompt = create_prompt(
        topic=topic,
        subtopic=subtopic,
        context=context,
        bloom=bloom,
        difficulty=difficulty,
        number_of_questions=candidate_count,
    )

    try:
        raw_response = call_ollama(prompt)

    except Exception as exc:
        print(f"[QWEN] Generation error: {exc}")
        return []

    parsed = extract_json(raw_response)

    questions = normalize_generated_questions(parsed)

    # Fallback if JSON parsing failed.
    if not questions:
        questions = normalize_generated_questions(raw_response)

    # ========================================================
    # UNDERSTAND-SPECIFIC FILTER
    # ========================================================

    if bloom == "Understand":
        filtered_questions = []

        for question in questions:
            if understand_question_is_valid(question):
                filtered_questions.append(question)

        questions = filtered_questions

    # Remove duplicates while preserving order.
    final_questions: List[str] = []

    seen = set()

    for question in questions:
        key = question.lower().strip()

        if key in seen:
            continue

        seen.add(key)
        final_questions.append(question)

    return final_questions[:number_of_questions]


# ============================================================
# REFINE QUESTION
# ============================================================

def refine_question_with_qwen(
    question: str,
    topic: str = "",
    context: str = "",
    bloom: str = "Understand",
    difficulty: str = "Medium",
    requested_bloom: Optional[str] = None,
    requested_difficulty: Optional[str] = None,
    **kwargs,
) -> str:

    if requested_bloom:
        bloom = requested_bloom

    if requested_difficulty:
        difficulty = requested_difficulty

    bloom = normalize_bloom(bloom)
    difficulty = normalize_difficulty(difficulty)

    bloom_instruction = get_bloom_instruction(bloom)
    difficulty_instruction = get_difficulty_instruction(difficulty)

    prompt = f"""
You are revising an engineering exam question.

TOPIC:
{topic}

REFERENCE CONTEXT:
{context}

CURRENT QUESTION:
{question}

TARGET BLOOM LEVEL:
{bloom}

TARGET DIFFICULTY:
{difficulty}

============================================================
BLOOM REQUIREMENT
============================================================

{bloom_instruction}

============================================================
DIFFICULTY REQUIREMENT
============================================================

{difficulty_instruction}

============================================================
REVISION RULES
============================================================

Rewrite the question so that it clearly matches the requested
Bloom level and difficulty.

For Understand questions:

Prefer:
- Explain how...
- Describe how...
- Explain why...
- Describe the process by which...

Avoid:
- What is...
- Define...
- Name...
- List...
- Which...
- What does...
- Compare...
- Contrast...
- Distinguish...

Keep the question concise and directly supported by the context.

Return ONLY the revised question.
Do not include an answer.
Do not include labels.
Do not include explanation.
"""

    try:
        response = call_ollama(prompt)

    except Exception as exc:
        print(f"[QWEN] Refinement error: {exc}")
        return clean_question(question)

    revised = clean_question(response)

    if not revised:
        return clean_question(question)

    if bloom == "Understand":
        if not understand_question_is_valid(revised):
            return clean_question(question)

    return revised


# ============================================================
# CLASSIFY QUESTION WITH QWEN
# ============================================================

def classify_question_with_llm(
    question: str,
    context: str = "",
    **kwargs,
) -> Dict[str, Any]:

    prompt = f"""
You are an expert educational assessment classifier.

Classify the following engineering question according to:

1. Bloom's taxonomy:
   Remember
   Understand
   Apply
   Analyze
   Evaluate
   Create

2. Difficulty:
   Easy
   Medium
   Hard

QUESTION:
{question}

REFERENCE CONTEXT:
{context}

============================================================
BLOOM CLASSIFICATION RULE
============================================================

Classify based on the ACTUAL cognitive task required by the
question, not merely the first word.

Remember:
Recall a fact, definition, term, component, or basic fact.

Understand:
Explain, describe, interpret, summarize, or explain how/why
a concept or process works.

Apply:
Use a known rule, algorithm, formula, or procedure on a
specific case.

Analyze:
Break information into parts, compare alternatives, identify
relationships, or examine differences requiring reasoning.

Evaluate:
Make and justify a judgment using criteria or evidence.

Create:
Design, construct, formulate, or propose something new.

IMPORTANT:

A question beginning with "Explain" or "Describe" is not
automatically Understand. Inspect what the student actually
has to do.

Likewise, "Compare", "Contrast", or "Distinguish" often indicates
Analyze when the student must deeply examine relationships.

============================================================
OUTPUT
============================================================

Return ONLY valid JSON:

{{
  "bloom": "Understand",
  "difficulty": "Easy",
  "confidence": 0.90
}}
"""

    try:
        raw_response = call_ollama(prompt)

    except Exception as exc:
        print(f"[QWEN] Classification error: {exc}")

        return {
            "bloom": "Understand",
            "difficulty": "Medium",
            "confidence": 0.0,
        }

    parsed = extract_json(raw_response)

    if not isinstance(parsed, dict):
        return {
            "bloom": "Understand",
            "difficulty": "Medium",
            "confidence": 0.0,
        }

    bloom = normalize_bloom(parsed.get("bloom"))
    difficulty = normalize_difficulty(parsed.get("difficulty"))

    try:
        confidence = float(parsed.get("confidence", 0.0))
    except Exception:
        confidence = 0.0

    confidence = max(
        0.0,
        min(1.0, confidence),
    )

    return {
        "bloom": bloom,
        "difficulty": difficulty,
        "confidence": confidence,
    }


# ============================================================
# COMPATIBILITY ALIASES
# ============================================================

def classify_question(
    question: str,
    context: str = "",
    **kwargs,
) -> Dict[str, Any]:

    return classify_question_with_llm(
        question=question,
        context=context,
        **kwargs,
    )


def generate_question(
    topic: str,
    subtopic: str = "",
    context: str = "",
    bloom: str = "Understand",
    difficulty: str = "Medium",
    **kwargs,
) -> str:

    questions = generate_questions(
        topic=topic,
        subtopic=subtopic,
        context=context,
        number_of_questions=1,
        bloom=bloom,
        difficulty=difficulty,
        **kwargs,
    )

    if not questions:
        return ""

    return questions[0]
