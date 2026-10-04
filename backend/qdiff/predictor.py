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

MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 128


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"QDiff device: {DEVICE}")


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

state_dict = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

model.load_state_dict(state_dict)

model.to(DEVICE)
model.eval()

print("QDiff model loaded successfully.")


# ============================================================
# PREDICT QUESTION
# ============================================================

def predict_question(question: str):

    if not question or not question.strip():
        return {
            "success": False,
            "error": "Question cannot be empty."
        }

    # --------------------------------------------------------
    # Tokenize
    # --------------------------------------------------------

    encoded = tokenizer(
        question,
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH,
        return_tensors="pt"
    )

    input_ids = encoded["input_ids"].to(DEVICE)
    attention_mask = encoded["attention_mask"].to(DEVICE)

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    with torch.no_grad():

        bloom_logits, difficulty_logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # Convert logits to probabilities
        bloom_probs = torch.softmax(
            bloom_logits,
            dim=-1
        )[0]

        difficulty_probs = torch.softmax(
            difficulty_logits,
            dim=-1
        )[0]

    # --------------------------------------------------------
    # Get predictions
    # --------------------------------------------------------

    bloom_id = torch.argmax(bloom_probs).item()
    difficulty_id = torch.argmax(difficulty_probs).item()

    bloom_level = BLOOM_LABELS[bloom_id]
    difficulty = DIFFICULTY_LABELS[difficulty_id]

    bloom_confidence = float(
        bloom_probs[bloom_id].item()
    )

    difficulty_confidence = float(
        difficulty_probs[difficulty_id].item()
    )

    # Overall confidence
    confidence = (
        bloom_confidence +
        difficulty_confidence
    ) / 2

    # --------------------------------------------------------
    # Probability breakdown
    # --------------------------------------------------------

    difficulty_probabilities = {
        "Easy": round(
            float(difficulty_probs[0].item()),
            4
        ),
        "Medium": round(
            float(difficulty_probs[1].item()),
            4
        ),
        "Hard": round(
            float(difficulty_probs[2].item()),
            4
        )
    }

    bloom_probabilities = {
        BLOOM_LABELS[i]: round(
            float(bloom_probs[i].item()),
            4
        )
        for i in range(len(BLOOM_LABELS))
    }

    # --------------------------------------------------------
    # Result
    # --------------------------------------------------------

    return {
        "success": True,

        "question": question,

        "bloom_level": bloom_level,
        "difficulty": difficulty,

        "bloom_confidence": round(
            bloom_confidence,
            4
        ),

        "difficulty_confidence": round(
            difficulty_confidence,
            4
        ),

        "confidence": round(
            confidence,
            4
        ),

        # NEW
        "difficulty_probabilities":
            difficulty_probabilities,

        "bloom_probabilities":
            bloom_probabilities
    }