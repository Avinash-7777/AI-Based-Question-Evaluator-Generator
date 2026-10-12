import sqlite3
import json
from pathlib import Path
from datetime import datetime, timezone

DB_PATH = Path(__file__).resolve().parent / "feedback.db"


def get_connection():
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def initialize_feedback_db():
    with get_connection() as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                feature TEXT NOT NULL,
                rating TEXT NOT NULL,
                reason TEXT,
                comment TEXT,
                question TEXT,
                predicted_bloom TEXT,
                corrected_bloom TEXT,
                predicted_difficulty TEXT,
                corrected_difficulty TEXT,
                result_id TEXT,
                created_at TEXT NOT NULL
            )
        """)


def save_feedback(data):
    with get_connection() as connection:
        cursor = connection.execute("""
            INSERT INTO feedback (
                feature, rating, reason, comment, question,
                predicted_bloom, corrected_bloom,
                predicted_difficulty, corrected_difficulty,
                result_id, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            data.get("feature"),
            data.get("rating"),
            data.get("reason"),
            data.get("comment"),
            data.get("question"),
            data.get("predicted_bloom"),
            data.get("corrected_bloom"),
            data.get("predicted_difficulty"),
            data.get("corrected_difficulty"),
            data.get("result_id"),
            datetime.now(timezone.utc).isoformat()
        ))

        return cursor.lastrowid


def get_feedback_summary():
    with get_connection() as connection:
        total = connection.execute(
            "SELECT COUNT(*) FROM feedback"
        ).fetchone()[0]

        ratings = connection.execute("""
            SELECT rating, COUNT(*) AS count
            FROM feedback
            GROUP BY rating
        """).fetchall()

        corrected_examples = connection.execute("""
            SELECT feature, question, predicted_bloom, corrected_bloom,
                   predicted_difficulty, corrected_difficulty
            FROM feedback
            WHERE corrected_bloom IS NOT NULL
               OR corrected_difficulty IS NOT NULL
            ORDER BY id DESC
            LIMIT 20
        """).fetchall()

        common_issues = connection.execute("""
            SELECT reason, COUNT(*) AS count
            FROM feedback
            WHERE reason IS NOT NULL AND reason != ''
            GROUP BY reason
            ORDER BY count DESC
            LIMIT 10
        """).fetchall()

    rating_counts = {
        "helpful": 0,
        "partly_helpful": 0,
        "not_helpful": 0
    }

    for row in ratings:
        rating_counts[row["rating"]] = row["count"]

    return {
        "total_feedback": total,
        "ratings": rating_counts,
        "corrected_examples": [
            dict(row) for row in corrected_examples
        ],
        "common_issues": [
            dict(row) for row in common_issues
        ]
    }


initialize_feedback_db()
