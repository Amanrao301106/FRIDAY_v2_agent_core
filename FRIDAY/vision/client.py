from __future__ import annotations

import base64
import json
from abc import ABC, abstractmethod
from pathlib import Path


class VisionBackend(ABC):
    """Interface for a screen/image vision backend."""

    @abstractmethod
    def analyze(self, image_path: str, instruction: str) -> str:
        """Analyze an image and return the model's raw response."""
        raise NotImplementedError


class OllamaVisionClient(VisionBackend):
    """Vision backend using a local Ollama vision model."""

    def __init__(
        self,
        model: str = "llava",
        host: str = "http://localhost:11434",
    ):
        self.model = model
        self.host = host

        # IMPORTANT:
        # host belongs on the Ollama Client, not on client.chat().
        from ollama import Client

        self.client = Client(host=host)

    def analyze(self, image_path: str, instruction: str) -> str:
        image_path = Path(image_path)

        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        image_data = base64.b64encode(
            image_path.read_bytes()
        ).decode("utf-8")

        response = self.client.chat(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": instruction,
                    "images": [image_data],
                }
            ],
        )

        return response["message"]["content"]


def parse_target_response(raw: str):
    """
    Parse JSON returned by the vision model.

    Handles:
    1. Pure JSON
    2. JSON surrounded by normal text
    """
    raw = raw.strip()

    # Case 1: response is already valid JSON.
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Case 2: model added text around the JSON.
    start = raw.find("{")
    end = raw.rfind("}")

    if start != -1 and end != -1 and end > start:
        candidate = raw[start:end + 1]

        try:
            return json.loads(candidate)
        except json.JSONDecodeError as exc:
            raise ValueError(
                "Vision model returned invalid JSON."
            ) from exc

    raise ValueError(
        "No valid JSON object found in vision response."
    )