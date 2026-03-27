# ============================================================
# 🌙 Multi-LLM Router
#
# Intelligent routing across multiple AI providers.
# Priority order:
#   1. DeepSeek  — primary (cheapest, best for code)
#   2. Groq      — free tier fallback (fast, llama-3.3-70b)
#   3. Gemini    — secondary fallback (free tier)
#   4. OpenAI    — last resort
#
# Automatically falls back if primary is:
#   - Out of credits (402)
#   - Rate limited (429)
#   - Down / timeout
#
# HOW TO USE (drop-in replacement for deepseek_model.py):
#   from src.models.llm_router import model
#   response = model.chat(system_prompt, user_prompt)
#
# Add keys to .env:
#   DEEPSEEK_API_KEY=...   (platform.deepseek.com)
#   GROQ_API_KEY=...       (console.groq.com — FREE)
#   GEMINI_KEY=...         (aistudio.google.com — FREE)
#   OPENAI_KEY=...         (optional)
# ============================================================

import os
import time
import traceback
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# ── Provider configs ──────────────────────────────────────────
PROVIDERS = [
    {
        "name":     "DeepSeek",
        "env_key":  "DEEPSEEK_API_KEY",
        "base_url": "https://api.deepseek.com",
        "model":    "deepseek-chat",
        "emoji":    "🧠",
    },
    {
        "name":     "Groq",
        "env_key":  "GROQ_API_KEY",
        "base_url": "https://api.groq.com/openai/v1",
        "model":    "llama-3.3-70b-versatile",
        "emoji":    "⚡",
    },
    {
        "name":     "Gemini",
        "env_key":  "GEMINI_KEY",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "model":    "gemini-2.0-flash",
        "emoji":    "💎",
    },
    {
        "name":     "OpenAI",
        "env_key":  "OPENAI_KEY",
        "base_url": "https://api.openai.com/v1",
        "model":    "gpt-4o-mini",
        "emoji":    "🤖",
    },
]

# Error codes that mean we should try next provider
FALLBACK_STATUS_CODES = {402, 429, 500, 502, 503, 504}


class LLMRouter:
    """
    Smart multi-provider LLM router with automatic fallback.
    Tries providers in order until one succeeds.
    """

    def __init__(self):
        self.clients     = {}   # provider name → OpenAI client
        self.available   = []   # providers with valid API keys
        self.last_used   = None
        self.fail_counts = {}   # track failures per provider

        for p in PROVIDERS:
            key = os.getenv(p["env_key"])
            if key:
                try:
                    client = OpenAI(api_key=key, base_url=p["base_url"])
                    self.clients[p["name"]] = (client, p)
                    self.available.append(p["name"])
                    self.fail_counts[p["name"]] = 0
                except Exception:
                    pass

        if not self.available:
            raise EnvironmentError(
                "No AI provider keys found.\n"
                "Add at least one to your .env:\n"
                "  DEEPSEEK_API_KEY  → platform.deepseek.com\n"
                "  GROQ_API_KEY      → console.groq.com (FREE)\n"
                "  GEMINI_KEY        → aistudio.google.com (FREE)\n"
            )

        print(f"🔀 LLM Router ready — providers: {' → '.join(self.available)}")

    def chat(self, system_prompt: str, user_prompt: str,
             max_tokens: int = 4096, prefer: str = None) -> str:
        """
        Send a prompt and return response.
        Automatically falls back to next provider on failure.

        prefer: optionally force a specific provider name
        """
        # Build attempt order
        order = list(self.available)
        if prefer and prefer in order:
            order.remove(prefer)
            order.insert(0, prefer)

        last_error = None

        for name in order:
            client, config = self.clients[name]
            try:
                response = client.chat.completions.create(
                    model=config["model"],
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user",   "content": user_prompt},
                    ],
                    max_tokens=max_tokens,
                    temperature=0.1,
                )
                result = response.choices[0].message.content.strip()

                # Reset fail count on success
                self.fail_counts[name] = 0
                self.last_used = name

                # Only log if we had to fall back
                if name != self.available[0]:
                    print(f"  {config['emoji']} [{name}] responded (fallback from {self.available[0]})")

                return result

            except Exception as e:
                err_str    = str(e)
                err_code   = getattr(e, 'status_code', 0)
                self.fail_counts[name] = self.fail_counts.get(name, 0) + 1
                last_error = e

                # Check if this is a known fallback-worthy error
                should_fallback = (
                    err_code in FALLBACK_STATUS_CODES
                    or "insufficient" in err_str.lower()
                    or "rate limit"   in err_str.lower()
                    or "quota"        in err_str.lower()
                    or "timeout"      in err_str.lower()
                    or "connection"   in err_str.lower()
                    or "402"          in err_str
                    or "429"          in err_str
                )

                if should_fallback:
                    next_providers = [p for p in order if p != name]
                    if next_providers:
                        print(f"  ⚠️  [{name}] failed ({err_code or err_str[:40]})"
                              f" → trying {next_providers[0]}...")
                    continue
                else:
                    # Non-recoverable error — re-raise immediately
                    raise

        # All providers failed
        raise RuntimeError(
            f"All LLM providers failed. Last error: {last_error}\n"
            f"Providers tried: {order}"
        )

    def status(self) -> dict:
        """Return current status of all providers."""
        return {
            name: {
                "available": name in self.available,
                "fail_count": self.fail_counts.get(name, 0),
                "last_used":  name == self.last_used,
                "model":      self.clients[name][1]["model"] if name in self.clients else "—",
            }
            for name in [p["name"] for p in PROVIDERS]
        }

    def print_status(self):
        """Print a nice status table."""
        print("\n🔀 LLM Router Status:")
        for name, info in self.status().items():
            if info["available"]:
                active = " ← active" if info["last_used"] else ""
                print(f"   ✅ {name:<12} {info['model']:<30} fails:{info['fail_count']}{active}")
            else:
                print(f"   ❌ {name:<12} (no API key)")
        print()


# ── Singleton ─────────────────────────────────────────────────
model = LLMRouter()


# ── Backwards compatibility ───────────────────────────────────
# All existing agents import `from src.models.llm_router import model`
# This file is the drop-in replacement. Update imports or add alias:
# from src.models.llm_router import model
