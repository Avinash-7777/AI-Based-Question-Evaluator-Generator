import torch
import torch.nn as nn
from transformers import AutoModel


class QDiffModel(nn.Module):

    def __init__(
        self,
        model_name="distilbert-base-uncased"
    ):
        super().__init__()

        self.encoder = AutoModel.from_pretrained(
            model_name
        )

        hidden_size = self.encoder.config.hidden_size

        # Bloom's Taxonomy: 6 classes
        self.bloom_classifier = nn.Linear(
            hidden_size,
            6
        )

        # Difficulty: 3 classes
        self.difficulty_classifier = nn.Linear(
            hidden_size,
            3
        )

    def forward(
        self,
        input_ids,
        attention_mask
    ):

        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # CLS representation
        pooled_output = outputs.last_hidden_state[:, 0]

        bloom_logits = self.bloom_classifier(
            pooled_output
        )

        difficulty_logits = self.difficulty_classifier(
            pooled_output
        )

        return (
            bloom_logits,
            difficulty_logits
        )