# src/fatingest/vlm.py

import os
import base64
import httpx
from pathlib import Path
from openai import OpenAI

# Config fra environment
BASE_URL = os.getenv("VLM_BASE_URL", "http://localhost:11434/v1")
API_KEY = os.getenv("VLM_API_KEY", "ollama")
MODEL = os.getenv("VLM_MODEL", "qwen2.5-vl:7b")
TIMEOUT = float(os.getenv("VLM_TIMEOUT", "300")) # 5 min default for VLM

client = OpenAI(
    base_url=BASE_URL,
    api_key=API_KEY,
    timeout=httpx.Timeout(TIMEOUT, connect=10.0), # long read timeout, short connect
)


def encode_image(image_path: str | Path) -> tuple[str, str]:
    """Returnerer (base64_data, media_type)"""
    path = Path(image_path)
    suffix = path.suffix.lower()
    media_types = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp",
    }
    media_type = media_types.get(suffix, "image/jpeg")
    
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8"), media_type


def analyze_image(image_path: str | Path, prompt: str = "Beskriv dette billede.") -> str:
    base64_data, media_type = encode_image(image_path)
    
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{media_type};base64,{base64_data}"},
                },
            ],
        }],
        max_tokens=1024,
    )
    return response.choices[0].message.content