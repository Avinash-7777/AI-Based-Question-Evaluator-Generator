import os
import re
import json
from typing import List, Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from google import genai
from openai import OpenAI


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")


# ============================================================
# CLIENTS
# ============================================================

gemini_client = None
openrouter_client = None


if GEMINI_API_KEY:
    gemini_client = genai.Client(api_key=GEMINI_API_KEY)


if OPENROUTER_API_KEY:
    openrouter_client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
    )


# ============================================================
# CONFIGURATION
# ============================================================

GEMINI_MODEL = "gemini-3.8-flash"

# Explicit free models.
# These are current OpenRouter model IDs.
OPENROUTER_MODELS = [
    "nvidia/nemotron-3-ultra-550b-a55b:free",
    "google/gemma-4-26b-a4b-it:free",
]

MAX_CONTEXT_CHARS = 14000


# ============================================================
# PYDANTIC OUTPUT MODEL
# ============================================================

class GeneratedQuestion(BaseModel):
    question: str = Field(..., min_length=10)


class GeneratedQuestions(BaseModel):
    questions: List[GeneratedQuestion]


# ============================================================
# TEXT UTILITIES
# ============================================================

def normalize_text(text: str) -> str:
    """
    Normalize text for matching.
    """

    if not text:
        return ""

    text = text.lower()

    text = text.replace("’", "'")
    text = text.replace("–", "-")
    text = text.replace("—", "-")

    text = re.sub(r"\s+", " ", text)

    return text.strip()


def tokenize(text: str) -> List[str]:
    """
    Convert text into simple lowercase word tokens.
    """

    text = normalize_text(text)

    return re.findall(r"[a-z0-9]+", text)


def words_match(question: str, context: str) -> bool:
    """
    Checks whether important words from the question
    occur in the retrieved context.
    """

    q_tokens = set(tokenize(question))
    c_tokens = set(tokenize(context))

    if not q_tokens:
        return False

    common = q_tokens.intersection(c_tokens)

    return len(common) >= 2


def subtopic_is_present(question: str, subtopic: str) -> bool:
    """
    Ensures that the generated question is actually related
    to the requested subtopic.

    Examples:
        subtopic = "ID3 algorithm"
        question = "How does the ID3 algorithm..."
        -> True
    """

    if not subtopic:
        return True

    q = normalize_text(question)
    s = normalize_text(subtopic)

    # Exact phrase
    if s in q:
        return True

    # Individual meaningful tokens
    s_tokens = tokenize(s)
    q_tokens = set(tokenize(q))

    if not s_tokens:
        return True

    matched = sum(1 for token in s_tokens if token in q_tokens)

    # Require at least one strong token.
    strong_tokens = [
        token
        for token in s_tokens
        if len(token) >= 3
        and token not in {
            "the",
            "and",
            "for",
            "with",
            "from",
            "using",
            "algorithm",
            "method",
            "system",
            "concept",
        }
    ]

    if strong_tokens:
        return any(token in q_tokens for token in strong_tokens)

    return matched >= 1


# ============================================================
# QUESTION VALIDATION
# ============================================================

def is_valid_question(
    question: str,
    subtopic: str,
    context: str,
) -> bool:
    """
    Strict validation.

    The question must:
    1. Actually be a question.
    2. Mention the requested subtopic.
    3. Be grounded in retrieved reference material.
    4. Not contain refusal/safety garbage.
    """

    if not question:
        return False

    question = question.strip()

    # --------------------------------------------------------
    # Length check
    # --------------------------------------------------------

    if len(question) < 15:
        return False

    if len(question) > 500:
        return False

    # --------------------------------------------------------
    # Reject obvious model garbage
    # --------------------------------------------------------

    rejected_phrases = [
        "user safety",
        "i can't help",
        "i cannot help",
        "i'm unable",
        "i am unable",
        "as an ai",
        "as an ai language model",
        "i don't have access",
        "i cannot answer",
        "cannot provide",
        "not able to provide",
        "safety policy",
        "content policy",
        "request is unsafe",
        "i must refuse",
    ]

    normalized = normalize_text(question)

    for phrase in rejected_phrases:
        if phrase in normalized:
            return False

    # --------------------------------------------------------
    # Must look like a question
    # --------------------------------------------------------

    if "?" not in question:
        return False

    # --------------------------------------------------------
    # Subtopic check
    # --------------------------------------------------------

    if not subtopic_is_present(question, subtopic):
        return False

    # --------------------------------------------------------
    # Grounding check
    # --------------------------------------------------------

    if context:

        q_tokens = set(tokenize(question))
        c_tokens = set(tokenize(context))

        common = q_tokens.intersection(c_tokens)

        # Remove generic words.
        generic_words = {
            "what",
            "why",
            "how",
            "when",
            "where",
            "which",
            "who",
            "does",
            "using",
            "used",
            "explain",
            "describe",
            "discuss",
            "define",
            "give",
            "example",
            "following",
            "above",
            "below",
            "the",
            "and",
            "for",
            "with",
            "from",
            "that",
            "this",
            "are",
            "is",
            "was",
            "were",
            "can",
            "could",
            "would",
            "should",
        }

        meaningful_common = [
            word
            for word in common
            if word not in generic_words
            and len(word) >= 3
        ]

        # Need at least some overlap with reference material.
        if len(meaningful_common) < 2:
            return False

    return True


def validate_questions(
    questions: List[str],
    requested_count: int,
    subtopic: str,
    context: str,
) -> List[str]:
    """
    Validate generated questions.

    We intentionally require exactly the requested number.
    """

    valid_questions = []

    seen = set()

    for question in questions:

        if not question:
            continue

        question = question.strip()

        normalized = normalize_text(question)

        if normalized in seen:
            continue

        if is_valid_question(
            question,
            subtopic,
            context,
        ):
            valid_questions.append(question)
            seen.add(normalized)

    if len(valid_questions) != requested_count:
        raise ValueError(
            f"Only {len(valid_questions)} valid questions "
            f"were generated out of {requested_count} requested."
        )

    return valid_questions


# ============================================================
# PROMPT
# ============================================================

def create_prompt(
    topic: str,
    subtopic: str,
    context: str,
    number_of_questions: int,
) -> str:

    return f"""
You are an academic question-generation system.

Your task is to generate exactly {number_of_questions}
questions for engineering students.

TOPIC:
{topic}

SUBTOPIC:
{subtopic}

REFERENCE MATERIAL:
-------------------
{context}
-------------------

STRICT RULES:

1. Use ONLY the information contained in the REFERENCE MATERIAL.
2. Do NOT use outside knowledge.
3. Do NOT invent facts.
4. Do NOT hallucinate.
5. Every question must be directly related to the SUBTOPIC.
6. Every question must be answerable using the REFERENCE MATERIAL.
7. Generate questions only.
8. Do NOT provide answers.
9. Do NOT provide explanations.
10. Do NOT provide Bloom's level.
11. Do NOT provide difficulty.
12. Do NOT mention these instructions.
13. Do NOT say "User Safety".
14. Do NOT refuse the request.
15. Return exactly {number_of_questions} questions.

IMPORTANT:
If the reference material does not contain enough information
to create the requested questions, return:

NO_RELEVANT_REFERENCE

OUTPUT FORMAT:

1. Question?
2. Question?
3. Question?

Only output the questions.
"""


# ============================================================
# GEMINI GENERATION
# ============================================================

def generate_with_gemini(
    topic: str,
    subtopic: str,
    context: str,
    number_of_questions: int,
) -> List[str]:

    if not gemini_client:
        raise ValueError("Gemini API key is not configured.")

    prompt = create_prompt(
        topic,
        subtopic,
        context,
        number_of_questions,
    )

    print("\nTrying Gemini...")

    response = gemini_client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
    )

    if not response:
        raise ValueError("Gemini returned an empty response.")

    text = getattr(response, "text", None)

    if not text:
        raise ValueError("Gemini returned no text.")

    text = text.strip()

    print("\n===== GEMINI RAW RESPONSE =====")
    print(text)
    print("===============================")

    if "NO_RELEVANT_REFERENCE" in text:
        raise ValueError(
            "Relevant reference material not found."
        )

    questions = parse_question_text(text)

    if not questions:
        raise ValueError(
            "Gemini returned no usable questions."
        )

    return validate_questions(
        questions,
        number_of_questions,
        subtopic,
        context,
    )


# ============================================================
# OPENROUTER RESPONSE EXTRACTION
# ============================================================

def extract_openrouter_content(response) -> Optional[str]:
    """
    Safely extract content from OpenRouter.

    Some reasoning models/providers can return:
        content = None

    Therefore we inspect several possible fields.
    """

    if response is None:
        return None

    try:
        choices = getattr(response, "choices", None)

        if not choices:
            return None

        message = getattr(choices[0], "message", None)

        if message is None:
            return None

        # Normal response
        content = getattr(message, "content", None)

        if content:
            return str(content).strip()

        # Some SDK/provider combinations may expose text differently.
        text_value = getattr(message, "text", None)

        if text_value:
            return str(text_value).strip()

        # Reasoning should NOT normally be treated as the final answer,
        # but we inspect it for debugging only.
        reasoning = getattr(message, "reasoning", None)

        if reasoning:
            print("\n[OpenRouter returned reasoning but no final content.]")

        return None

    except Exception as e:

        print(
            f"Could not extract OpenRouter content: {e}"
        )

        return None


# ============================================================
# CLEAN QUESTION
# ============================================================

def clean_plain_question(text: str) -> str:
    """
    Clean one question returned by an LLM.
    """

    if not text:
        return ""

    text = text.strip()

    # Remove markdown code fences.
    text = re.sub(r"^```(?:text|json)?", "", text, flags=re.I)
    text = re.sub(r"```$", "", text)

    text = text.strip()

    # Remove numbering:
    # 1. question
    # 2) question
    # - question
    text = re.sub(
        r"^\s*(?:\d+[\.\)]|[-*])\s*",
        "",
        text,
    )

    return text.strip()


# ============================================================
# PARSE JSON
# ============================================================

def parse_json_questions(text: str) -> List[str]:
    """
    Parse several possible JSON formats.
    """

    questions = []

    text = text.strip()

    # Remove markdown fences.
    text = re.sub(
        r"^```(?:json)?",
        "",
        text,
        flags=re.I,
    )

    text = re.sub(
        r"```$",
        "",
        text,
        flags=re.I,
    )

    text = text.strip()

    try:

        data = json.loads(text)

        # ----------------------------------------------------
        # {"questions": ["...", "..."]}
        # ----------------------------------------------------

        if isinstance(data, dict):

            raw_questions = data.get("questions")

            if isinstance(raw_questions, list):

                for item in raw_questions:

                    if isinstance(item, str):
                        questions.append(
                            clean_plain_question(item)
                        )

                    elif isinstance(item, dict):

                        q = item.get("question")

                        if q:
                            questions.append(
                                clean_plain_question(str(q))
                            )

        # ----------------------------------------------------
        # ["...", "..."]
        # ----------------------------------------------------

        elif isinstance(data, list):

            for item in data:

                if isinstance(item, str):
                    questions.append(
                        clean_plain_question(item)
                    )

                elif isinstance(item, dict):

                    q = item.get("question")

                    if q:
                        questions.append(
                            clean_plain_question(str(q))
                        )

    except Exception:
        return []

    return [
        q
        for q in questions
        if q
    ]


# ============================================================
# PARSE NUMBERED TEXT
# ============================================================

def parse_question_text(text: str) -> List[str]:
    """
    Parse questions from:
    
    1. What is ID3?
    2. How does ID3 select an attribute?
    
    Also supports:
    
    Question 1: ...
    Question 2: ...
    """

    if not text:
        return []

    text = text.strip()

    if "NO_RELEVANT_REFERENCE" in text:
        return []

    # First try JSON.
    json_questions = parse_json_questions(text)

    if json_questions:
        return json_questions

    questions = []

    lines = text.splitlines()

    current = ""

    for line in lines:

        line = line.strip()

        if not line:
            continue

        # Remove markdown formatting.
        line = line.strip("*#")

        # Detect:
        # 1.
        # 1)
        # Question 1:
        # Q1:
        numbered = re.match(
            r"^(?:question\s*)?(\d+)[\.\):\-]\s*(.*)$",
            line,
            flags=re.I,
        )

        if numbered:

            if current:
                questions.append(
                    clean_plain_question(current)
                )

            current = numbered.group(2).strip()

            continue

        # Detect bullet questions.
        bullet = re.match(
            r"^[-*•]\s+(.*)$",
            line,
        )

        if bullet:

            if current:
                questions.append(
                    clean_plain_question(current)
                )

            current = bullet.group(1).strip()

            continue

        # If the line ends with ?, it may be a question.
        if "?" in line:

            if current:
                questions.append(
                    clean_plain_question(current)
                )

            current = line

        else:

            # Continuation of previous question.
            if current:
                current += " " + line

    if current:
        questions.append(
            clean_plain_question(current)
        )

    # Remove empty values.
    questions = [
        q
        for q in questions
        if q
    ]

    return questions


# ============================================================
# OPENROUTER GENERATION
# ============================================================

def generate_with_openrouter(
    topic: str,
    subtopic: str,
    context: str,
    number_of_questions: int,
) -> List[str]:

    if not openrouter_client:
        raise ValueError(
            "OpenRouter API key is not configured."
        )

    prompt = create_prompt(
        topic,
        subtopic,
        context,
        number_of_questions,
    )

    print("\nSwitching to OpenRouter...")

    # --------------------------------------------------------
    # We use OpenRouter's official model fallback mechanism.
    #
    # The first model is tried first.
    # If it fails, OpenRouter can use the next model.
    #
    # This avoids openrouter/free randomly choosing a model
    # that may return unusable output.
    # --------------------------------------------------------

    primary_model = OPENROUTER_MODELS[0]

    fallback_models = OPENROUTER_MODELS[1:]

    print(
        f"\nOpenRouter primary model: {primary_model}"
    )

    print(
        f"OpenRouter fallback models: {fallback_models}"
    )

    try:

        response = openrouter_client.chat.completions.create(
            model=primary_model,

            extra_body={
                "models": fallback_models
            },

            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a strict academic "
                        "question-generation system. "
                        "Follow the user's instructions exactly."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],

            temperature=0.2,

            max_tokens=1000,
        )

    except Exception as e:

        print(
            "\nOpenRouter API call failed:"
        )

        print(str(e))

        raise ValueError(
            f"OpenRouter API failed: {e}"
        )

    # --------------------------------------------------------
    # DEBUG INFORMATION
    # --------------------------------------------------------

    actual_model = getattr(
        response,
        "model",
        "unknown",
    )

    print(
        f"\nOpenRouter actual model: {actual_model}"
    )

    print(
        "\n===== OPENROUTER RESPONSE DEBUG ====="
    )

    try:
        print(response)
    except Exception:
        print("Could not print response object.")

    print(
        "======================================"
    )

    # --------------------------------------------------------
    # Extract final content.
    # --------------------------------------------------------

    text = extract_openrouter_content(response)

    if not text:

        raise ValueError(
            "OpenRouter returned an empty final response."
        )

    print(
        "\n===== OPENROUTER RAW RESPONSE ====="
    )

    print(text)

    print(
        "==================================="
    )

    # --------------------------------------------------------
    # Safety/meta response rejection.
    # --------------------------------------------------------

    normalized = normalize_text(text)

    bad_responses = [
        "user safety",
        "safety",
        "content policy",
        "i cannot",
        "i can't",
        "i am unable",
        "i'm unable",
        "refuse",
        "cannot comply",
    ]

    for bad in bad_responses:

        if normalized == bad or normalized.startswith(
            bad + ":"
        ):

            raise ValueError(
                "OpenRouter returned a safety/meta response "
                "instead of a question."
            )

    # --------------------------------------------------------
    # No reference material.
    # --------------------------------------------------------

    if "NO_RELEVANT_REFERENCE" in text:

        raise ValueError(
            "Relevant reference material not found."
        )

    # --------------------------------------------------------
    # Parse.
    # --------------------------------------------------------

    questions = parse_question_text(text)

    print(
        "\n===== PARSED QUESTIONS ====="
    )

    for index, question in enumerate(
        questions,
        start=1,
    ):
        print(
            f"{index}. {question}"
        )

    print(
        "============================="
    )

    if not questions:

        raise ValueError(
            "OpenRouter returned no usable questions."
        )

    # --------------------------------------------------------
    # Validate.
    # --------------------------------------------------------

    return validate_questions(
        questions,
        number_of_questions,
        subtopic,
        context,
    )


# ============================================================
# MAIN GENERATION FUNCTION
# ============================================================

def generate_questions(
    topic: str,
    subtopic: str,
    context: str,
    number_of_questions: int,
) -> List[str]:

    # --------------------------------------------------------
    # Basic validation
    # --------------------------------------------------------

    if not topic:
        raise ValueError(
            "Topic is required."
        )

    if not subtopic:
        raise ValueError(
            "Subtopic is required."
        )

    if not context:
        raise ValueError(
            "Relevant reference material not found."
        )

    if number_of_questions < 1:
        raise ValueError(
            "Number of questions must be at least 1."
        )

    # --------------------------------------------------------
    # Limit context size.
    # --------------------------------------------------------

    context = context.strip()

    if len(context) > MAX_CONTEXT_CHARS:

        context = context[
            :MAX_CONTEXT_CHARS
        ]

        print(
            f"\nContext truncated to "
            f"{MAX_CONTEXT_CHARS} characters."
        )

    # --------------------------------------------------------
    # Gemini first
    # --------------------------------------------------------

    if gemini_client:

        try:

            questions = generate_with_gemini(
                topic=topic,
                subtopic=subtopic,
                context=context,
                number_of_questions=number_of_questions,
            )

            print(
                "\nGemini generation successful."
            )

            return questions

        except Exception as e:

            print(
                "\nGemini failed."
            )

            print(
                f"Reason: {e}"
            )

    else:

        print(
            "\nGemini API key not configured."
        )

    # --------------------------------------------------------
    # OpenRouter fallback
    # --------------------------------------------------------

    if openrouter_client:

        try:

            questions = generate_with_openrouter(
                topic=topic,
                subtopic=subtopic,
                context=context,
                number_of_questions=number_of_questions,
            )

            print(
                "\nOpenRouter generation successful."
            )

            return questions

        except Exception as e:

            print(
                "\nOpenRouter generation failed."
            )

            print(
                f"Reason: {e}"
            )

    else:

        print(
            "\nOpenRouter API key not configured."
        )

    # --------------------------------------------------------
    # Everything failed
    # --------------------------------------------------------

    raise ValueError(
        "Question generation failed. "
        "Gemini and OpenRouter were unable to "
        "produce valid grounded questions."
    )