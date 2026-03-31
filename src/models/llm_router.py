# ============================================================
# 🌙 Multi-LLM Router
#
# Intelligent routing across multiple AI providers.
#
# DEFAULT order (all agents except RBI):
#   1. DeepSeek  — primary (cheapest, fast)
#   2. Groq      — free fallback
#   3. Gemini    — secondary fallback
#   4. OpenAI    — last resort
#
# RBI BACKTESTER order (strategy research + code generation):
#   1. Claude    — primary (best code quality, understands ICT)
#   2. DeepSeek  — fallback
#   3. Groq      — fallback
#
# Usage:
#   from src.models.llm_router import model       # default (DeepSeek first)
#   from src.models.llm_router import rbi_model   # RBI (Claude first)
#
# Add to .env:
#   ANTHROPIC_API_KEY=sk-ant-...   (claude.ai/settings or console.anthropic.com)
#   DEEPSEEK_API_KEY=...
#   GROQ_API_KEY=...
#   GEMINI_KEY=...
# ============================================================

import os
import time
import traceback
from dotenv import load_dotenv

load_dotenv()

# ── OpenAI-compatible providers ───────────────────────────────
from openai import OpenAI

OAI_PROVIDERS = [
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

FALLBACK_STATUS_CODES = {400, 402, 429, 500, 502, 503, 504}  # 400 = Claude credit error


class LLMRouter:
    """
    Smart multi-provider LLM router with automatic fallback.

    claude_first: if True, uses Claude as primary (for RBI backtester).
                  if False, uses DeepSeek as primary (for all other agents).
    """

    def __init__(self, claude_first: bool = False):
        self.clients     = {}
        self.available   = []
        self.last_used   = None
        self.fail_counts = {}
        self.claude_first = claude_first

        # ── Claude (Anthropic native API) ────────────────────
        anthropic_key = os.getenv("ANTHROPIC_API_KEY")
        if anthropic_key:
            try:
                import anthropic as _anthropic
                self._anthropic_client = _anthropic.Anthropic(api_key=anthropic_key)
                self.clients["Claude"] = ("anthropic", {
                    "name":  "Claude",
                    "model": "claude-sonnet-4-5",
                    "emoji": "✨",
                })
                self.fail_counts["Claude"] = 0
                if claude_first:
                    self.available.insert(0, "Claude")
                else:
                    self.available.append("Claude")
            except ImportError:
                print("  ⚠️  anthropic package not installed — run: pip install anthropic")
            except Exception as e:
                print(f"  ⚠️  Claude init failed: {e}")
        else:
            self._anthropic_client = None

        # ── OpenAI-compatible providers ───────────────────────
        for p in OAI_PROVIDERS:
            key = os.getenv(p["env_key"])
            if key:
                try:
                    client = OpenAI(api_key=key, base_url=p["base_url"])
                    self.clients[p["name"]] = (client, p)
                    self.fail_counts[p["name"]] = 0
                    if p["name"] not in self.available:
                        self.available.append(p["name"])
                except Exception:
                    pass

        if not self.available:
            raise EnvironmentError(
                "No AI provider keys found in .env\n"
                "Add at least one:\n"
                "  ANTHROPIC_API_KEY → console.anthropic.com\n"
                "  DEEPSEEK_API_KEY  → platform.deepseek.com\n"
                "  GROQ_API_KEY      → console.groq.com (FREE)\n"
            )

        primary = self.available[0]
        print(f"🔀 LLM Router ready [{'RBI mode' if claude_first else 'default'}]"
              f" — {' → '.join(self.available)}")

    def _chat_claude(self, system_prompt: str, user_prompt: str,
                     max_tokens: int) -> str:
        """Call Claude via native Anthropic API."""
        _, config = self.clients["Claude"]
        response = self._anthropic_client.messages.create(
            model=config["model"],
            max_tokens=max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return response.content[0].text.strip()

    def _chat_oai(self, name: str, system_prompt: str,
                  user_prompt: str, max_tokens: int) -> str:
        """Call an OpenAI-compatible provider."""
        client, config = self.clients[name]
        response = client.chat.completions.create(
            model=config["model"],
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            max_tokens=max_tokens,
            temperature=0.1,
        )
        return response.choices[0].message.content.strip()

    def chat(self, system_prompt: str, user_prompt: str,
             max_tokens: int = 4096, prefer: str = None) -> str:
        """
        Send a prompt and return response.
        Automatically falls back to next provider on failure.
        """
        order = list(self.available)
        if prefer and prefer in order:
            order.remove(prefer)
            order.insert(0, prefer)

        last_error = None

        for name in order:
            try:
                if name == "Claude":
                    result = self._chat_claude(system_prompt, user_prompt, max_tokens)
                else:
                    result = self._chat_oai(name, system_prompt, user_prompt, max_tokens)

                self.fail_counts[name] = 0
                self.last_used = name

                if name != self.available[0]:
                    config = self.clients[name]
                    emoji  = config[1]["emoji"] if isinstance(config, tuple) else "✨"
                    print(f"  {emoji} [{name}] responded (fallback)")

                return result

            except Exception as e:
                err_str  = str(e)
                err_code = getattr(e, "status_code", 0)
                self.fail_counts[name] = self.fail_counts.get(name, 0) + 1
                last_error = e

                should_fallback = (
                    err_code in FALLBACK_STATUS_CODES
                    or "insufficient" in err_str.lower()
                    or "rate limit"   in err_str.lower()
                    or "quota"        in err_str.lower()
                    or "timeout"      in err_str.lower()
                    or "connection"   in err_str.lower()
                    or "overloaded"   in err_str.lower()
                    or "credit balance" in err_str.lower()
                    or "too low"      in err_str.lower()
                    or "billing"      in err_str.lower()
                    or "400" in err_str
                    or "402" in err_str
                    or "429" in err_str
                )

                if should_fallback:
                    next_p = [p for p in order if p != name]
                    if next_p:
                        print(f"  ⚠️  [{name}] failed → trying {next_p[0]}...")
                    continue
                else:
                    raise

        raise RuntimeError(
            f"All LLM providers failed. Last error: {last_error}\n"
            f"Providers tried: {order}"
        )

    def status(self) -> dict:
        return {
            name: {
                "available":  name in self.available,
                "fail_count": self.fail_counts.get(name, 0),
                "last_used":  name == self.last_used,
                "model":      (self.clients[name][1]["model"]
                               if name in self.clients else "—"),
            }
            for name in list(self.clients.keys())
        }

    def print_status(self):
        print("\n🔀 LLM Router Status:")
        for name, info in self.status().items():
            if info["available"]:
                active = " ← active" if info["last_used"] else ""
                print(f"   ✅ {name:<12} {info['model']:<35}"
                      f" fails:{info['fail_count']}{active}")
            else:
                print(f"   ❌ {name:<12} (no API key)")
        print()


# ── Singletons ────────────────────────────────────────────────
# Default router — DeepSeek first (all agents except RBI)
model = LLMRouter(claude_first=False)

# RBI router — Claude first (strategy research + code generation)
# Used by rbi_parallel.py and rbi_agent.py
try:
    rbi_model = LLMRouter(claude_first=True)
except Exception:
    # Fall back to default if Claude not configured
    rbi_model = model
    print("  ℹ️  RBI model falling back to default (add ANTHROPIC_API_KEY for Claude)")