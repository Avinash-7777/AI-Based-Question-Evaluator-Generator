"""Review and export corrected QGen AI feedback as candidate training examples.

Place this file beside main.py and feedback_store.py.
This module does NOT train a model or modify question-generation behavior.
"""

import json
import sqlite3
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Dict, List

DB_PATH = Path(__file__).resolve().parent / "feedback.db"
EXPORT_PATH = Path(__file__).resolve().parent / "approved_feedback_training.jsonl"

ALLOWED_BLOOM = {"Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create"}
ALLOWED_DIFFICULTY = {"Easy", "Medium", "Hard"}


def _connect():
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def initialize_learning_review() -> None:
    """Add review metadata to an existing feedback table without deleting data."""
    if not DB_PATH.exists():
        raise RuntimeError(
            "feedback.db was not found. Start the backend once so feedback_store.py can create it."
        )

    with _connect() as connection:
        columns = {
            row["name"]
            for row in connection.execute("PRAGMA table_info(feedback)").fetchall()
        }
        if not columns:
            raise RuntimeError(
                "The feedback table was not found. Check feedback_store.py and feedback.db."
            )
        if "learning_approved" not in columns:
            connection.execute(
                "ALTER TABLE feedback ADD COLUMN learning_approved INTEGER NOT NULL DEFAULT 0"
            )
        if "learning_reviewed_at" not in columns:
            connection.execute(
                "ALTER TABLE feedback ADD COLUMN learning_reviewed_at TEXT"
            )


def list_learning_candidates() -> List[Dict[str, Any]]:
    """Return corrected feedback awaiting review, excluding ratings-only entries."""
    initialize_learning_review()
    with _connect() as connection:
        rows = connection.execute(
            """
            SELECT id, feature, rating, reason, comment, question,
                   predicted_bloom, corrected_bloom,
                   predicted_difficulty, corrected_difficulty, created_at
            FROM feedback
            WHERE COALESCE(learning_approved, 0) = 0
              AND TRIM(COALESCE(question, '')) <> ''
              AND (corrected_bloom IS NOT NULL OR corrected_difficulty IS NOT NULL)
            ORDER BY id DESC
            LIMIT 500
            """
        ).fetchall()
    return [dict(row) for row in rows]


def review_learning_candidate(feedback_id: int, approve: bool) -> Dict[str, Any]:
    """Approve or reject a correction for training export; this does not train a model."""
    initialize_learning_review()
    with _connect() as connection:
        row = connection.execute(
            """SELECT id, question, corrected_bloom, corrected_difficulty
               FROM feedback WHERE id = ?""",
            (feedback_id,),
        ).fetchone()

        if row is None:
            raise ValueError("Feedback record was not found.")
        if not str(row["question"] or "").strip():
            raise ValueError("This feedback has no question text and cannot be used as a training example.")

        if approve:
            bloom = row["corrected_bloom"]
            difficulty = row["corrected_difficulty"]
            if bloom is not None and bloom not in ALLOWED_BLOOM:
                raise ValueError("The corrected Bloom level is invalid.")
            if difficulty is not None and difficulty not in ALLOWED_DIFFICULTY:
                raise ValueError("The corrected difficulty is invalid.")
            if bloom is None and difficulty is None:
                raise ValueError("There is no corrected label to approve.")

        reviewed_at = datetime.now(timezone.utc).isoformat()
        with connection:
            connection.execute(
                "UPDATE feedback SET learning_approved = ?, learning_reviewed_at = ? WHERE id = ?",
                (1 if approve else -1, reviewed_at, feedback_id),
            )

    return {
        "feedback_id": feedback_id,
        "review_status": "approved" if approve else "rejected",
        "reviewed_at": reviewed_at,
    }


def export_approved_training_examples() -> Dict[str, Any]:
    """Export only manually approved, corrected examples to JSON Lines."""
    initialize_learning_review()
    examples = []
    with _connect() as connection:
        rows = connection.execute(
            """
            SELECT id, feature, question, corrected_bloom, corrected_difficulty
            FROM feedback
            WHERE learning_approved = 1
              AND TRIM(COALESCE(question, '')) <> ''
              AND (corrected_bloom IS NOT NULL OR corrected_difficulty IS NOT NULL)
            ORDER BY id ASC
            """
        ).fetchall()

    for row in rows:
        bloom = row["corrected_bloom"]
        difficulty = row["corrected_difficulty"]
        if bloom is not None and bloom not in ALLOWED_BLOOM:
            continue
        if difficulty is not None and difficulty not in ALLOWED_DIFFICULTY:
            continue
        examples.append({
            "feedback_id": row["id"],
            "feature": row["feature"],
            "text": row["question"].strip(),
            "bloom": bloom,
            "difficulty": difficulty,
        })

    with EXPORT_PATH.open("w", encoding="utf-8") as file:
        for example in examples:
            file.write(json.dumps(example, ensure_ascii=False) + "\n")

    return {"count": len(examples), "path": str(EXPORT_PATH)}
