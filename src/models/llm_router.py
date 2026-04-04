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
        # llama-3.3-70b-versatile is Groq's best model
        # Falls back automatically within Groq if rate limited
        "model":    "llama-3.3-70b-versatile",
        "emoji":    "⚡",
    },
    {
        "name":     "Gemini",
        "env_key":  "GEMINI_KEY",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        # gemini-2.0-flash is free tier — limit: 1500 req/day
        # If hitting quota, upgrade to paid or use gemini-1.5-flash-latest
        "model":    "gemini-1.5-flash-latest",
        "emoji":    "💎",
    },
    {
        "name":     "OpenAI",
        "env_key":  "OPENAI_KEY",
        "base_url": "https://api.openai.com/v1",
        "model":    "gpt-4o-mini",
        "emoji":    "🤖",
    },
    {
        # ── OpenRouter — 29+ free models, one API key ──────────
        # Sign up FREE at openrouter.ai → no credit card needed
        # Best free coding model: Qwen3 Coder 480B (rivals Claude)
        # API is OpenAI-compatible — drop-in replacement
        # Rate limits: 20 req/min, 200 req/day (resets daily)
        # Get key at: openrouter.ai → API Keys → Create Key
        "name":     "OpenRouter",
        "env_key":  "OPENROUTER_API_KEY",
        "base_url": "https://openrouter.ai/api/v1",
        "model":    "qwen/qwen3-coder:free",   # best free coding model
        "emoji":    "🌐",
    },
]

# OpenRouter free model cascade — tried in order when primary is rate-limited
OPENROUTER_FREE_MODELS = [
    "qwen/qwen3-coder:free",             # #1 — Qwen3 Coder 480B, best for code
    "qwen/qwen3.6-plus-preview:free",    # #2 — 1M context, strong reasoning
    "deepseek/deepseek-r1:free",         # #3 — DeepSeek R1 (reasoning model)
    "openrouter/free",                   # #4 — auto-selects best available free model
    "meta-llama/llama-3.3-70b-instruct:free",  # #5 — Llama 3.3 70B
    "openai/gpt-oss-120b:free",          # #6 — OpenAI open-weight 120B
    "mistralai/mistral-small-3.1:free",  # #7 — Mistral Small 3.1
    "google/gemma-3-27b-it:free",        # #8 — Google Gemma 3 27B
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
        # Claude Sonnet has 200k context but RBI prompts can be large.
        # Cap max_tokens at 8192 to prevent 400 errors on long prompts.
        safe_max = min(max_tokens, 8192)
        # Also truncate extremely long prompts (RBI code generation)
        combined_len = len(system_prompt) + len(user_prompt)
        if combined_len > 150_000:
            # Trim user_prompt to fit — keep system prompt intact
            trim_to = 150_000 - len(system_prompt) - 1000
            user_prompt = user_prompt[:trim_to] + "\n[...truncated for length...]"
        response = self._anthropic_client.messages.create(
            model=config["model"],
            max_tokens=safe_max,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return response.content[0].text.strip()

    def _chat_oai(self, name: str, system_prompt: str,
                  user_prompt: str, max_tokens: int) -> str:
        """Call an OpenAI-compatible provider."""
        client, config = self.clients[name]

        # OpenRouter: rotate through free models if primary is rate-limited
        if name == "OpenRouter":
            last_err = None
            for model_name in OPENROUTER_FREE_MODELS:
                try:
                    response = client.chat.completions.create(
                        model=model_name,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user",   "content": user_prompt},
                        ],
                        max_tokens=max_tokens,
                        temperature=0.1,
                    )
                    if model_name != OPENROUTER_FREE_MODELS[0]:
                        print(f"    ↳ OpenRouter: using {model_name}")
                    return response.choices[0].message.content.strip()
                except Exception as e:
                    last_err = e
                    err_s = str(e)
                    if "429" in err_s or "rate" in err_s.lower() or "quota" in err_s.lower():
                        continue   # try next free model
                    raise          # different error — propagate
            raise last_err

        # Groq: if primary model rate-limited, try smaller model
        groq_fallback_models = [
            "llama-3.3-70b-versatile",    # Groq primary — best quality
            "llama-3.1-8b-instant",       # faster, less likely rate-limited
            "llama3-70b-8192",            # alternate 70B
            "llama3-8b-8192",             # smallest — almost never rate-limited
        ]
        # gemma2-9b-it and mixtral-8x7b removed — discontinued on Groq

        models_to_try = (
            groq_fallback_models if name == "Groq"
            else [config["model"]]
        )

        last_err = None
        for model_name in models_to_try:
            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user",   "content": user_prompt},
                    ],
                    max_tokens=max_tokens,
                    temperature=0.1,
                )
                if model_name != config["model"]:
                    print(f"    ↳ Groq: using {model_name} (primary rate-limited)")
                return response.choices[0].message.content.strip()
            except Exception as e:
                last_err = e
                if "429" in str(e) or "rate" in str(e).lower():
                    continue   # try next Groq model
                raise          # other error — propagate

        raise last_err

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
                    # Find the NEXT provider in order (not just any other)
                    try:
                        current_idx = order.index(name)
                        remaining   = order[current_idx + 1:]
                    except ValueError:
                        remaining = []
                    if remaining:
                        print(f"  ⚠️  [{name}] failed ({type(e).__name__}: "
                              f"{str(e)[:60]}) → trying {remaining[0]}...")
                    else:
                        print(f"  ⚠️  [{name}] failed — all providers exhausted")
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