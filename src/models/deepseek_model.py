# ============================================================
# 🌙 Model Factory — DeepSeek AI Client
# ============================================================

import os
from openai import OpenAI   # DeepSeek uses OpenAI-compatible API
from dotenv import load_dotenv

load_dotenv()


class DeepSeekModel:
    """Thin wrapper around the DeepSeek chat API."""

    BASE_URL = "https://api.deepseek.com"
    MODEL    = "deepseek-chat"

    def __init__(self):
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "DEEPSEEK_API_KEY not found. "
                "Add it to your .env file — get one free at platform.deepseek.com"
            )
        self.client = OpenAI(api_key=api_key, base_url=self.BASE_URL)

    def chat(self, system_prompt: str, user_prompt: str, max_tokens: int = 4096) -> str:
        """Send a prompt and return the text response."""
        response = self.client.chat.completions.create(
            model=self.MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            max_tokens=max_tokens,
            temperature=0.1,   # low = more deterministic code
        )
        return response.choices[0].message.content.strip()


# Singleton — import and reuse across agents
model = DeepSeekModel()
