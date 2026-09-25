import os
import pandas as pd
import torch

from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from torch.optim import AdamW

from model import QDiffModel
from labels import BLOOM_TO_ID, DIFFICULTY_TO_ID


# ============================================================
# SETTINGS
# ============================================================

MODEL_NAME = "distilbert-base-uncased"

TRAIN_FILE = "../qdiff_dataset/train.xlsx"
VAL_FILE = "../qdiff_dataset/validation.xlsx"

BATCH_SIZE = 8
EPOCHS = 3
LEARNING_RATE = 2e-5
MAX_LENGTH = 128

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", DEVICE)


# ============================================================
# DATASET
# ============================================================

class QuestionDataset(Dataset):

    def __init__(self, dataframe, tokenizer):

        self.data = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        row = self.data.iloc[index]

        question = str(row["Question"])

        encoding = self.tokenizer(
            question,
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
            return_tensors="pt"
        )

        bloom_label = BLOOM_TO_ID[
            str(row["Bloom's Level"]).strip()
        ]

        difficulty_label = DIFFICULTY_TO_ID[
            str(row["Difficulty Level"]).strip()
        ]

        return {
            "input_ids": encoding["input_ids"].squeeze(0),

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

train_df = pd.read_excel(TRAIN_FILE)
val_df = pd.read_excel(VAL_FILE)

print("Training samples:", len(train_df))
print("Validation samples:", len(val_df))


# ============================================================
# TOKENIZER
# ============================================================

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)


# ============================================================
# DATA LOADERS
# ============================================================

train_dataset = QuestionDataset(
    train_df,
    tokenizer
)

val_dataset = QuestionDataset(
    val_df,
    tokenizer
)

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


# ============================================================
# MODEL
# ============================================================

model = QDiffModel(
    model_name=MODEL_NAME
)

model.to(DEVICE)


# ============================================================
# CLASS-WEIGHTED LOSS
# ============================================================

# Bloom has six classes.
# We keep equal weights for Bloom.

bloom_weights = torch.tensor(
    [
        1.0,  # Remember
        1.0,  # Understand
        1.0,  # Apply
        1.0,  # Analyze
        1.0,  # Evaluate
        1.0   # Create
    ],
    dtype=torch.float32
).to(DEVICE)


# Difficulty classes:
#
# 0 = Easy
# 1 = Medium
# 2 = Hard
#
# Hard receives a higher weight.

difficulty_weights = torch.tensor(
    [
        1.0,   # Easy
        0.75,  # Medium
        1.5    # Hard
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
# TRAINING
# ============================================================

for epoch in range(EPOCHS):

    print(
        f"\nStarting Epoch "
        f"{epoch + 1}/{EPOCHS}..."
    )

    model.train()

    total_train_loss = 0

    bloom_correct = 0
    difficulty_correct = 0

    total_samples = 0


    # --------------------------------------------------------
    # TRAINING LOOP
    # --------------------------------------------------------

    for batch_index, batch in enumerate(
        train_loader
    ):

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


        optimizer.zero_grad()


        # Forward pass

        bloom_logits, difficulty_logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )


        # Bloom loss

        bloom_loss = bloom_criterion(
            bloom_logits,
            bloom_labels
        )


        # Difficulty loss

        difficulty_loss = difficulty_criterion(
            difficulty_logits,
            difficulty_labels
        )


        # Multi-task loss

        loss = (
            bloom_loss +
            difficulty_loss
        )


        # Backpropagation

        loss.backward()

        optimizer.step()


        total_train_loss += loss.item()


        # ----------------------------------------------------
        # TRAINING ACCURACY
        # ----------------------------------------------------

        bloom_predictions = torch.argmax(
            bloom_logits,
            dim=1
        )

        difficulty_predictions = torch.argmax(
            difficulty_logits,
            dim=1
        )


        bloom_correct += (
            bloom_predictions == bloom_labels
        ).sum().item()


        difficulty_correct += (
            difficulty_predictions ==
            difficulty_labels
        ).sum().item()


        total_samples += (
            bloom_labels.size(0)
        )


        # ----------------------------------------------------
        # PROGRESS
        # ----------------------------------------------------

        if (batch_index + 1) % 20 == 0:

            print(
                f"Epoch {epoch + 1} "
                f"- Batch "
                f"{batch_index + 1}/"
                f"{len(train_loader)} "
                f"- Loss: "
                f"{loss.item():.4f}"
            )


    # --------------------------------------------------------
    # TRAINING RESULTS
    # --------------------------------------------------------

    average_train_loss = (
        total_train_loss /
        len(train_loader)
    )


    train_bloom_accuracy = (
        100 *
        bloom_correct /
        total_samples
    )


    train_difficulty_accuracy = (
        100 *
        difficulty_correct /
        total_samples
    )


    print(
        f"\nTraining Loss: "
        f"{average_train_loss:.4f}"
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

    total_val_loss = 0

    val_bloom_correct = 0
    val_difficulty_correct = 0

    val_samples = 0


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


            # Forward pass

            bloom_logits, difficulty_logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )


            # Validation losses

            bloom_loss = bloom_criterion(
                bloom_logits,
                bloom_labels
            )

            difficulty_loss = difficulty_criterion(
                difficulty_logits,
                difficulty_labels
            )


            loss = (
                bloom_loss +
                difficulty_loss
            )


            total_val_loss += loss.item()


            # Predictions

            bloom_predictions = torch.argmax(
                bloom_logits,
                dim=1
            )

            difficulty_predictions = torch.argmax(
                difficulty_logits,
                dim=1
            )


            # Correct predictions

            val_bloom_correct += (
                bloom_predictions ==
                bloom_labels
            ).sum().item()


            val_difficulty_correct += (
                difficulty_predictions ==
                difficulty_labels
            ).sum().item()


            val_samples += (
                bloom_labels.size(0)
            )


    # --------------------------------------------------------
    # VALIDATION RESULTS
    # --------------------------------------------------------

    average_val_loss = (
        total_val_loss /
        len(val_loader)
    )


    val_bloom_accuracy = (
        100 *
        val_bloom_correct /
        val_samples
    )


    val_difficulty_accuracy = (
        100 *
        val_difficulty_correct /
        val_samples
    )


    print(
        f"Validation Loss: "
        f"{average_val_loss:.4f}"
    )

    print(
        f"Validation Bloom Accuracy: "
        f"{val_bloom_accuracy:.2f}%"
    )

    print(
        f"Validation Difficulty Accuracy: "
        f"{val_difficulty_accuracy:.2f}%"
    )


    print(
        f"\nEpoch {epoch + 1}/{EPOCHS} "
        f"completed."
    )


# ============================================================
# SAVE MODEL
# ============================================================

os.makedirs(
    "../qdiff_model",
    exist_ok=True
)


torch.save(
    model.state_dict(),
    "../qdiff_model/qdiff_model.pt"
)


tokenizer.save_pretrained(
    "../qdiff_model"
)


print(
    "\nQDiff model saved successfully!"
)