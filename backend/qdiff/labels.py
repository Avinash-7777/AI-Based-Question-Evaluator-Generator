BLOOM_LABELS = [
    "Remember",
    "Understand",
    "Apply",
    "Analyze",
    "Evaluate",
    "Create"
]

DIFFICULTY_LABELS = [
    "Easy",
    "Medium",
    "Hard"
]


BLOOM_TO_ID = {
    label: i
    for i, label in enumerate(BLOOM_LABELS)
}


DIFFICULTY_TO_ID = {
    label: i
    for i, label in enumerate(DIFFICULTY_LABELS)
}