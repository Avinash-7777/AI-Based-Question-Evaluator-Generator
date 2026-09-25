from pathlib import Path

import torch
from transformers import AutoTokenizer

from .model import QDiffModel
from .labels import BLOOM_LABELS, DIFFICULTY_LABELS


# ============================================================
# PATHS
# ============================================================

QDIFF_DIR = Path(__file__).resolve().parent

MODEL_DIR = QDIFF_DIR.parent / "qdiff_model"

MODEL_PATH = MODEL_DIR / "qdiff_model.pt"


# ============================================================
# SETTINGS
# ============================================================

MODEL_NAME = "distilbert-base-uncased"

MAX_LENGTH = 128

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


print("QDiff device:", DEVICE)


# ============================================================
# LOAD TOKENIZER
# ============================================================

tokenizer = AutoTokenizer.from_pretrained(
    str(MODEL_DIR)
)


# ============================================================
# LOAD MODEL
# ============================================================

model = QDiffModel(
    model_name=MODEL_NAME
)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )
)

model.to(DEVICE)

model.eval()

print("QDiff model loaded successfully.")


# ============================================================
# PREDICT
# ============================================================

def predict_question(question: str):

    encoding = tokenizer(
        question,
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
        return_tensors="pt"
    )

    input_ids = encoding[
        "input_ids"
    ].to(DEVICE)

    attention_mask = encoding[
        "attention_mask"
    ].to(DEVICE)


    with torch.no_grad():

        bloom_logits, difficulty_logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )


        # ----------------------------------------------------
        # Convert logits to probabilities
        # ----------------------------------------------------

        bloom_probabilities = torch.softmax(
            bloom_logits,
            dim=1
        )

        difficulty_probabilities = torch.softmax(
            difficulty_logits,
            dim=1
        )


        # ----------------------------------------------------
        # Get predicted classes
        # ----------------------------------------------------

        bloom_id = torch.argmax(
            bloom_probabilities,
            dim=1
        ).item()

        difficulty_id = torch.argmax(
            difficulty_probabilities,
            dim=1
        ).item()


        # ----------------------------------------------------
        # Confidence
        # ----------------------------------------------------

        bloom_confidence = bloom_probabilities[
            0,
            bloom_id
        ].item()

        difficulty_confidence = (
            difficulty_probabilities[
                0,
                difficulty_id
            ].item()
        )


    return {

        "bloom_level": BLOOM_LABELS[
            bloom_id
        ],

        "difficulty": DIFFICULTY_LABELS[
            difficulty_id
        ],

        "bloom_confidence": (
            bloom_confidence
        ),

        "difficulty_confidence": (
            difficulty_confidence
        ),

        "confidence": (
            bloom_confidence +
            difficulty_confidence
        ) / 2
    }


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    question = (
        "Why does pruning help prevent "
        "overfitting in decision trees?"
    )

    result = predict_question(
        question
    )

    print("\n===== QDIFF PREDICTION =====\n")

    print("Question:")
    print(question)

    print("\nBloom's Level:")
    print(result["bloom_level"])

    print(
        "Bloom Confidence:",
        f"{result['bloom_confidence'] * 100:.2f}%"
    )

    print("\nDifficulty:")
    print(result["difficulty"])

    print(
        "Difficulty Confidence:",
        f"{result['difficulty_confidence'] * 100:.2f}%"
    )

    print(
        "\nOverall Confidence:",
        f"{result['confidence'] * 100:.2f}%"
    )