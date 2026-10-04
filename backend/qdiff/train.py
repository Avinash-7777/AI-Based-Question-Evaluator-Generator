import os
import pandas as pd
import torch

from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW

from model import QDiffModel
from labels import BLOOM_TO_ID, DIFFICULTY_TO_ID


# ============================================================
# SETTINGS
# ============================================================

MODEL_NAME = "distilbert-base-uncased"

from pathlib import Path

QDIFF_DIR = Path(__file__).resolve().parent

TRAIN_FILE = QDIFF_DIR.parent / "qdiff_dataset" / "train.xlsx"
VAL_FILE = QDIFF_DIR.parent / "qdiff_dataset" / "validation.xlsx"

MODEL_DIR = "qdiff_model"
MODEL_PATH = os.path.join(
    MODEL_DIR,
    "qdiff_model.pt"
)

BATCH_SIZE = 8

# Increased from 3 to 5 so the difficulty head has more
# opportunity to learn.
EPOCHS = 5

LEARNING_RATE = 2e-5

MAX_LENGTH = 128

# ------------------------------------------------------------
# Multi-task loss weights
#
# Bloom is already performing better.
# Difficulty needs more learning pressure.
# ------------------------------------------------------------

BLOOM_LOSS_WEIGHT = 0.5
DIFFICULTY_LOSS_WEIGHT = 1.5


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 60)
print("QDiff Training")
print("=" * 60)

print(
    "Using device:",
    DEVICE
)

print(
    "Bloom loss weight:",
    BLOOM_LOSS_WEIGHT
)

print(
    "Difficulty loss weight:",
    DIFFICULTY_LOSS_WEIGHT
)

print(
    "Epochs:",
    EPOCHS
)

print("=" * 60)


# ============================================================
# DATASET
# ============================================================

class QuestionDataset(Dataset):

    def __init__(
        self,
        dataframe,
        tokenizer
    ):

        self.data = dataframe.reset_index(
            drop=True
        )

        self.tokenizer = tokenizer

    def __len__(self):

        return len(self.data)

    def __getitem__(self, index):

        row = self.data.iloc[index]

        question = str(
            row["Question"]
        ).strip()

        # ----------------------------------------------------
        # Tokenization
        # ----------------------------------------------------

        encoding = self.tokenizer(
            question,
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
            return_tensors="pt"
        )

        # ----------------------------------------------------
        # Bloom label
        # ----------------------------------------------------

        bloom_text = str(
            row["Bloom's Level"]
        ).strip()

        bloom_label = BLOOM_TO_ID[
            bloom_text
        ]

        # ----------------------------------------------------
        # Difficulty label
        # ----------------------------------------------------

        difficulty_text = str(
            row["Difficulty Level"]
        ).strip()

        difficulty_label = DIFFICULTY_TO_ID[
            difficulty_text
        ]

        # ----------------------------------------------------
        # Return
        # ----------------------------------------------------

        return {
            "input_ids": encoding[
                "input_ids"
            ].squeeze(0),

            "attention_mask": encoding[
                "attention_mask"
            ].squeeze(0),

            "bloom_label": torch.tensor(
                bloom_label,
                dtype=torch.long
            ),

            "difficulty_label": torch.tensor(
                difficulty_label,
                dtype=torch.long
            )
        }


# ============================================================
# LOAD DATA
# ============================================================

train_df = pd.read_excel(
    TRAIN_FILE
)

val_df = pd.read_excel(
    VAL_FILE
)


print("\nDataset information")
print("-" * 60)

print(
    "Training samples:",
    len(train_df)
)

print(
    "Validation samples:",
    len(val_df)
)


# ============================================================
# SHOW LABEL DISTRIBUTION
# ============================================================

print("\nTraining difficulty distribution")

print(
    train_df[
        "Difficulty Level"
    ].value_counts()
)

print("\nValidation difficulty distribution")

print(
    val_df[
        "Difficulty Level"
    ].value_counts()
)


print("\nTraining Bloom distribution")

print(
    train_df[
        "Bloom's Level"
    ].value_counts()
)


# ============================================================
# TOKENIZER
# ============================================================

print("\nLoading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

print(
    "Tokenizer loaded successfully."
)


# ============================================================
# DATASETS
# ============================================================

train_dataset = QuestionDataset(
    train_df,
    tokenizer
)

val_dataset = QuestionDataset(
    val_df,
    tokenizer
)


# ============================================================
# DATA LOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


print(
    "\nTraining batches:",
    len(train_loader)
)

print(
    "Validation batches:",
    len(val_loader)
)


# ============================================================
# MODEL
# ============================================================

print("\nLoading QDiff model...")

model = QDiffModel(
    model_name=MODEL_NAME
)

model.to(DEVICE)

print(
    "QDiff model loaded."
)


# ============================================================
# LOSS FUNCTIONS
# ============================================================

# ------------------------------------------------------------
# Bloom
#
# 0 = Remember
# 1 = Understand
# 2 = Apply
# 3 = Analyze
# 4 = Evaluate
# 5 = Create
#
# Equal weights intentionally retained.
# ------------------------------------------------------------

bloom_weights = torch.tensor(
    [
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0
    ],
    dtype=torch.float32
).to(DEVICE)


# ------------------------------------------------------------
# Difficulty
#
# 0 = Easy
# 1 = Medium
# 2 = Hard
#
# IMPORTANT:
# We are NOT using class weighting here because your previous
# class-balanced experiment reduced validation performance.
# ------------------------------------------------------------

difficulty_weights = torch.tensor(
    [
        1.0,
        1.0,
        1.0
    ],
    dtype=torch.float32
).to(DEVICE)


bloom_criterion = torch.nn.CrossEntropyLoss(
    weight=bloom_weights
)

difficulty_criterion = torch.nn.CrossEntropyLoss(
    weight=difficulty_weights
)


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = AdamW(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# LEARNING RATE SCHEDULER
# ============================================================

total_training_steps = (
    len(train_loader)
    * EPOCHS
)

warmup_steps = int(
    total_training_steps * 0.10
)

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_training_steps
)


print("\nTraining configuration")
print("-" * 60)

print(
    "Learning rate:",
    LEARNING_RATE
)

print(
    "Total training steps:",
    total_training_steps
)

print(
    "Warmup steps:",
    warmup_steps
)


# ============================================================
# BEST MODEL TRACKING
# ============================================================

best_difficulty_accuracy = 0.0
best_bloom_accuracy = 0.0

best_epoch = 0

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)


# ============================================================
# TRAINING
# ============================================================

for epoch in range(EPOCHS):

    print("\n")
    print("=" * 60)

    print(
        f"Epoch {epoch + 1}/{EPOCHS}"
    )

    print("=" * 60)


    # ========================================================
    # TRAIN MODE
    # ========================================================

    model.train()

    total_train_loss = 0.0

    total_bloom_loss = 0.0

    total_difficulty_loss = 0.0

    bloom_correct = 0

    difficulty_correct = 0

    total_samples = 0


    # ========================================================
    # TRAINING LOOP
    # ========================================================

    for batch_index, batch in enumerate(
        train_loader
    ):

        # ----------------------------------------------------
        # Move data to device
        # ----------------------------------------------------

        input_ids = batch[
            "input_ids"
        ].to(DEVICE)

        attention_mask = batch[
            "attention_mask"
        ].to(DEVICE)

        bloom_labels = batch[
            "bloom_label"
        ].to(DEVICE)

        difficulty_labels = batch[
            "difficulty_label"
        ].to(DEVICE)


        # ----------------------------------------------------
        # Clear gradients
        # ----------------------------------------------------

        optimizer.zero_grad()


        # ----------------------------------------------------
        # Forward pass
        # ----------------------------------------------------

        bloom_logits, difficulty_logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )


        # ----------------------------------------------------
        # Bloom loss
        # ----------------------------------------------------

        bloom_loss = bloom_criterion(
            bloom_logits,
            bloom_labels
        )


        # ----------------------------------------------------
        # Difficulty loss
        # ----------------------------------------------------

        difficulty_loss = difficulty_criterion(
            difficulty_logits,
            difficulty_labels
        )


        # ----------------------------------------------------
        # WEIGHTED MULTI-TASK LOSS
        #
        # Difficulty gets 3x the relative importance of Bloom.
        #
        # 0.5 Bloom + 1.5 Difficulty
        # ----------------------------------------------------

        loss = (
            BLOOM_LOSS_WEIGHT
            * bloom_loss
            +
            DIFFICULTY_LOSS_WEIGHT
            * difficulty_loss
        )


        # ----------------------------------------------------
        # Backpropagation
        # ----------------------------------------------------

        loss.backward()


        # ----------------------------------------------------
        # Gradient clipping
        # ----------------------------------------------------

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )


        # ----------------------------------------------------
        # Optimizer
        # ----------------------------------------------------

        optimizer.step()

        scheduler.step()


        # ----------------------------------------------------
        # Loss statistics
        # ----------------------------------------------------

        total_train_loss += loss.item()

        total_bloom_loss += (
            bloom_loss.item()
        )

        total_difficulty_loss += (
            difficulty_loss.item()
        )


        # ----------------------------------------------------
        # Predictions
        # ----------------------------------------------------

        bloom_predictions = torch.argmax(
            bloom_logits,
            dim=1
        )

        difficulty_predictions = torch.argmax(
            difficulty_logits,
            dim=1
        )


        # ----------------------------------------------------
        # Correct
        # ----------------------------------------------------

        bloom_correct += (
            bloom_predictions
            ==
            bloom_labels
        ).sum().item()

        difficulty_correct += (
            difficulty_predictions
            ==
            difficulty_labels
        ).sum().item()


        total_samples += (
            bloom_labels.size(0)
        )


        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if (
            batch_index + 1
        ) % 20 == 0:

            current_lr = optimizer.param_groups[
                0
            ]["lr"]

            print(
                f"Batch "
                f"{batch_index + 1}/"
                f"{len(train_loader)} "
                f"| Loss: "
                f"{loss.item():.4f} "
                f"| Bloom Loss: "
                f"{bloom_loss.item():.4f} "
                f"| Difficulty Loss: "
                f"{difficulty_loss.item():.4f} "
                f"| LR: "
                f"{current_lr:.2e}"
            )


    # ========================================================
    # TRAINING RESULTS
    # ========================================================

    average_train_loss = (
        total_train_loss
        /
        len(train_loader)
    )

    average_bloom_loss = (
        total_bloom_loss
        /
        len(train_loader)
    )

    average_difficulty_loss = (
        total_difficulty_loss
        /
        len(train_loader)
    )

    train_bloom_accuracy = (
        100.0
        *
        bloom_correct
        /
        total_samples
    )

    train_difficulty_accuracy = (
        100.0
        *
        difficulty_correct
        /
        total_samples
    )


    print("\nTraining Results")
    print("-" * 60)

    print(
        f"Training Loss: "
        f"{average_train_loss:.4f}"
    )

    print(
        f"Bloom Loss: "
        f"{average_bloom_loss:.4f}"
    )

    print(
        f"Difficulty Loss: "
        f"{average_difficulty_loss:.4f}"
    )

    print(
        f"Training Bloom Accuracy: "
        f"{train_bloom_accuracy:.2f}%"
    )

    print(
        f"Training Difficulty Accuracy: "
        f"{train_difficulty_accuracy:.2f}%"
    )


    # ========================================================
    # VALIDATION
    # ========================================================

    model.eval()

    total_val_loss = 0.0

    total_val_bloom_loss = 0.0

    total_val_difficulty_loss = 0.0

    val_bloom_correct = 0

    val_difficulty_correct = 0

    val_samples = 0


    # --------------------------------------------------------
    # Per-class difficulty statistics
    # --------------------------------------------------------

    difficulty_class_correct = {
        0: 0,
        1: 0,
        2: 0
    }

    difficulty_class_total = {
        0: 0,
        1: 0,
        2: 0
    }


    # --------------------------------------------------------
    # Validation loop
    # --------------------------------------------------------

    with torch.no_grad():

        for batch in val_loader:

            input_ids = batch[
                "input_ids"
            ].to(DEVICE)

            attention_mask = batch[
                "attention_mask"
            ].to(DEVICE)

            bloom_labels = batch[
                "bloom_label"
            ].to(DEVICE)

            difficulty_labels = batch[
                "difficulty_label"
            ].to(DEVICE)


            # ------------------------------------------------
            # Forward
            # ------------------------------------------------

            bloom_logits, difficulty_logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )


            # ------------------------------------------------
            # Losses
            # ------------------------------------------------

            bloom_loss = bloom_criterion(
                bloom_logits,
                bloom_labels
            )

            difficulty_loss = difficulty_criterion(
                difficulty_logits,
                difficulty_labels
            )


            loss = (
                BLOOM_LOSS_WEIGHT
                * bloom_loss
                +
                DIFFICULTY_LOSS_WEIGHT
                * difficulty_loss
            )


            total_val_loss += (
                loss.item()
            )

            total_val_bloom_loss += (
                bloom_loss.item()
            )

            total_val_difficulty_loss += (
                difficulty_loss.item()
            )


            # ------------------------------------------------
            # Predictions
            # ------------------------------------------------

            bloom_predictions = torch.argmax(
                bloom_logits,
                dim=1
            )

            difficulty_predictions = torch.argmax(
                difficulty_logits,
                dim=1
            )


            # ------------------------------------------------
            # Overall accuracy
            # ------------------------------------------------

            val_bloom_correct += (
                bloom_predictions
                ==
                bloom_labels
            ).sum().item()

            val_difficulty_correct += (
                difficulty_predictions
                ==
                difficulty_labels
            ).sum().item()


            val_samples += (
                bloom_labels.size(0)
            )


            # ------------------------------------------------
            # Difficulty per-class accuracy
            # ------------------------------------------------

            for class_id in range(3):

                mask = (
                    difficulty_labels
                    ==
                    class_id
                )

                class_total = mask.sum().item()

                if class_total > 0:

                    class_correct = (
                        difficulty_predictions[
                            mask
                        ]
                        ==
                        difficulty_labels[
                            mask
                        ]
                    ).sum().item()

                    difficulty_class_total[
                        class_id
                    ] += class_total

                    difficulty_class_correct[
                        class_id
                    ] += class_correct


    # ========================================================
    # VALIDATION RESULTS
    # ========================================================

    average_val_loss = (
        total_val_loss
        /
        len(val_loader)
    )

    average_val_bloom_loss = (
        total_val_bloom_loss
        /
        len(val_loader)
    )

    average_val_difficulty_loss = (
        total_val_difficulty_loss
        /
        len(val_loader)
    )

    val_bloom_accuracy = (
        100.0
        *
        val_bloom_correct
        /
        val_samples
    )

    val_difficulty_accuracy = (
        100.0
        *
        val_difficulty_correct
        /
        val_samples
    )


    print("\nValidation Results")
    print("-" * 60)

    print(
        f"Validation Loss: "
        f"{average_val_loss:.4f}"
    )

    print(
        f"Validation Bloom Loss: "
        f"{average_val_bloom_loss:.4f}"
    )

    print(
        f"Validation Difficulty Loss: "
        f"{average_val_difficulty_loss:.4f}"
    )

    print(
        f"Validation Bloom Accuracy: "
        f"{val_bloom_accuracy:.2f}%"
    )

    print(
        f"Validation Difficulty Accuracy: "
        f"{val_difficulty_accuracy:.2f}%"
    )


    # ========================================================
    # PER-CLASS DIFFICULTY RESULTS
    # ========================================================

    print("\nDifficulty Class Accuracy")
    print("-" * 60)

    difficulty_names = {
        0: "Easy",
        1: "Medium",
        2: "Hard"
    }


    for class_id in range(3):

        class_total = (
            difficulty_class_total[
                class_id
            ]
        )

        class_correct = (
            difficulty_class_correct[
                class_id
            ]
        )

        if class_total > 0:

            class_accuracy = (
                100.0
                *
                class_correct
                /
                class_total
            )

        else:

            class_accuracy = 0.0


        print(
            f"{difficulty_names[class_id]}: "
            f"{class_correct}/"
            f"{class_total} "
            f"("
            f"{class_accuracy:.2f}%"
            f")"
        )


    # ========================================================
    # SAVE BEST MODEL
    # ========================================================

    # Primary criterion:
    # difficulty validation accuracy.
    #
    # Tie breaker:
    # Bloom validation accuracy.

    should_save = False

    if (
        val_difficulty_accuracy
        >
        best_difficulty_accuracy
    ):

        should_save = True

    elif (
        val_difficulty_accuracy
        ==
        best_difficulty_accuracy
        and
        val_bloom_accuracy
        >
        best_bloom_accuracy
    ):

        should_save = True


    if should_save:

        best_difficulty_accuracy = (
            val_difficulty_accuracy
        )

        best_bloom_accuracy = (
            val_bloom_accuracy
        )

        best_epoch = epoch + 1


        torch.save(
            model.state_dict(),
            MODEL_PATH
        )

        tokenizer.save_pretrained(
            MODEL_DIR
        )


        print("\n*** BEST MODEL SAVED ***")

        print(
            "Epoch:",
            best_epoch
        )

        print(
            "Best difficulty accuracy:",
            f"{best_difficulty_accuracy:.2f}%"
        )

        print(
            "Best Bloom accuracy:",
            f"{best_bloom_accuracy:.2f}%"
        )

        print(
            "Path:",
            MODEL_PATH
        )


    print(
        f"\nEpoch {epoch + 1}/{EPOCHS} completed."
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)

print(
    "Best epoch:",
    best_epoch
)

print(
    "Best validation difficulty accuracy:",
    f"{best_difficulty_accuracy:.2f}%"
)

print(
    "Best validation Bloom accuracy:",
    f"{best_bloom_accuracy:.2f}%"
)

print(
    "Model:",
    MODEL_PATH
)

print("=" * 60)

print(
    "\nQDiff model saved successfully!"
)