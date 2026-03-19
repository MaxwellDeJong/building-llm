"""Training entrypoint.

Run directly:
    python train.py

Or via Docker:
    docker compose up
"""
import logging

from training.trainer import Trainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

trainer = Trainer(
    model_config_path="src/configs/gpt2_small.yaml",
    training_config_path="src/configs/training_config.yaml",
)
trainer.train()
