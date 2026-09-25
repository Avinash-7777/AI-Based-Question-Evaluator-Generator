import os
import time

from dotenv import load_dotenv
from google import genai
from openai import OpenAI
from pydantic import BaseModel


load_dotenv()


# ============================================================
# RESPONSE SCHEMA
# ============================================================

class GeneratedQuestion(BaseModel):
    question: str


class GeneratedQuestions(BaseModel):
    questions: list[GeneratedQuestion]


# ============================================================
# API KEYS
# ============================================================

gemini_api_key = os.getenv("GEMINI_API_KEY")
openrouter_api_key = os.getenv("OPENROUTER_API_KEY")


# ============================================================
# GEMINI CLIENT
# ============================================================

gemini_client = None

if gemini_api_key:
    gemini_client = genai.Client(
        api_key=gemini_api_key
    )


# ============================================================
# OPENROUTER CLIENT
# ============================================================

openrouter_client = None

if openrouter_api_key:
    openrouter_client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=openrouter_api_key
    )


# ============================================================
# PROMPT
# ============================================================

def create_prompt(
    topic: str,
    subtopic: str,
    context: str,
    number_of_questions: int
):

    return f"""
You are an educational question-generation system.

Generate {number_of_questions} high-quality academic
questions based ONLY on the provided educational context.

Topic:
{topic}

Subtopic:
{subtopic}

Educational Context:
{context}

Requirements:

1. Questions must be relevant to the topic.
2. Questions must be answerable using the provided context.
3. Avoid duplicate questions.
4. Questions should be suitable for engineering students.
5. Do not provide answers.
6. Do not include Bloom's level.
7. Do not include difficulty.
8. Generate exactly {number_of_questions} questions.
"""


# ============================================================
# GEMINI GENERATION
# ============================================================

def generate_with_gemini(prompt):

    if gemini_client is None:
        raise Exception(
            "Gemini API key is not configured."
        )

    response = gemini_client.models.generate_content(
        model="gemini-3.8-flash",
        contents=prompt,
        config={
            "response_mime_type": "application/json",
            "response_schema": GeneratedQuestions,
        },
    )

    result = response.parsed

    return [
        item.question
        for item in result.questions
    ]


# ============================================================
# OPENROUTER GENERATION
# ============================================================

def generate_with_openrouter(prompt):

    if openrouter_client is None:
        raise Exception(
            "OpenRouter API key is not configured."
        )

    response = openrouter_client.chat.completions.create(
        model="openrouter/free",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.7
    )

    content = response.choices[0].message.content

    return content


# ============================================================
# MAIN GENERATION FUNCTION
# ============================================================

def generate_questions(
    topic: str,
    subtopic: str,
    context: str,
    number_of_questions: int
):

    prompt = create_prompt(
        topic,
        subtopic,
        context,
        number_of_questions
    )

    # --------------------------------------------------------
    # TRY GEMINI FIRST
    # --------------------------------------------------------

    if gemini_client:

        try:

            print("\nTrying Gemini...")

            questions = generate_with_gemini(
                prompt
            )

            print("Gemini generation successful.")

            return questions

        except Exception as error:

            print(
                "\nGemini failed."
            )

            print(
                "Reason:",
                error
            )

            print(
                "\nSwitching to OpenRouter..."
            )


    # --------------------------------------------------------
    # FALLBACK TO OPENROUTER
    # --------------------------------------------------------

    if openrouter_client:

        try:

            questions_text = generate_with_openrouter(
                prompt
            )

            print(
                "OpenRouter generation successful."
            )

            # Try to extract numbered questions
            lines = questions_text.split("\n")

            questions = []

            for line in lines:

                line = line.strip()

                if not line:
                    continue

                # Remove common numbering formats
                if line[0].isdigit():

                    question = line

                    if "." in question[:4]:
                        question = question.split(
                            ".",
                            1
                        )[1].strip()

                    elif ")" in question[:4]:
                        question = question.split(
                            ")",
                            1
                        )[1].strip()

                    questions.append(question)

            # If parsing failed, return the raw response
            if not questions:

                questions = [
                    questions_text
                ]

            return questions[:number_of_questions]

        except Exception as error:

            print(
                "\nOpenRouter also failed."
            )

            raise error


    raise Exception(
        "No LLM API is configured."
    )