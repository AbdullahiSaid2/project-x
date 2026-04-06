# ============================================================
# Multi-LLM Router
#
# Intelligent routing across multiple AI providers.
#
# DEFAULT order (all agents except RBI):
# 1. Claude
# 2. DeepSeek
# 3. Groq
# 4. Gemini
# 5. OpenRouter
#
# RBI BACKTESTER order:
# 1. Claude
# 2. DeepSeek
# 3. Groq
# 4. Gemini
# 5. OpenRouter
#
# OpenRouter is handled via direct HTTP request so we explicitly
# control the Authorization header and parse provider-specific
# response formats safely.
# ============================================================

import os
from dotenv import load_dotenv

load_dotenv()

import requests
from openai import OpenAI

# ── OpenAI-compatible providers ───────────────────────────────
OAI_PROVIDERS = [
    {
        "name": "DeepSeek",
        "env_key": "DEEPSEEK_API_KEY",
        "base_url": "https://api.deepseek.com",
        "model": "deepseek-chat",
        "emoji": "",
    },
    {
        "name": "Groq",
        "env_key": "GROQ_API_KEY",
        "base_url": "https://api.groq.com/openai/v1",
        "model": "llama-3.3-70b-versatile",
        "emoji": "⚡",
    },
    {
        "name": "Gemini",
        "env_key": "GEMINI_KEY",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "model": "gemini-2.5-flash",
        "emoji": "",
    },
    {
        "name": "OpenAI",
        "env_key": "OPENAI_KEY",
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-4o-mini",
        "emoji": "",
    },
]

# Keep this conservative. openrouter/free is the stable entry point.
# Free model availability changes frequently on OpenRouter, so avoid
# depending on a long fragile list of explicit :free models.
OPENROUTER_MODELS = [
    "openrouter/free",
    "qwen/qwen3-coder:free",
    "meta-llama/llama-3.3-70b-instruct:free",
]

FALLBACK_STATUS_CODES = {400, 401, 402, 404, 408, 413, 429, 500, 502, 503, 504}


def _clean_env_value(value: str | None) -> str:
    if not value:
        return ""
    value = value.strip()
    if len(value) >= 2 and (
        (value[0] == '"' and value[-1] == '"')
        or (value[0] == "'" and value[-1] == "'")
    ):
        value = value[1:-1].strip()
    return value


def _get_openrouter_key() -> str:
    for name in ("OPENROUTER_API_KEY", "OPEN_ROUTER_API_KEY", "OPENROUTER_KEY"):
        value = _clean_env_value(os.getenv(name))
        if value:
            return value
    return ""


def _looks_like_openrouter_key(value: str) -> bool:
    return value.startswith("sk-or-")


class LLMRouter:
    def __init__(self, claude_first: bool = False):
        self.clients = {}
        self.available = []
        self.last_used = None
        self.fail_counts = {}
        self.claude_first = claude_first

        # ── Claude ───────────────────────────────────────────
        anthropic_key = _clean_env_value(os.getenv("ANTHROPIC_API_KEY"))
        if anthropic_key:
            try:
                import anthropic as _anthropic

                self._anthropic_client = _anthropic.Anthropic(api_key=anthropic_key)
                self.clients["Claude"] = (
                    "anthropic",
                    {
                        "name": "Claude",
                        "model": "claude-sonnet-4-5-20250929",
                        "emoji": "✨",
                    },
                )
                self.fail_counts["Claude"] = 0
                self.available.append("Claude")
            except ImportError:
                print("⚠️ anthropic package not installed — run: pip install anthropic")
                self._anthropic_client = None
            except Exception as e:
                print(f"⚠️ Claude init failed: {e}")
                self._anthropic_client = None
        else:
            self._anthropic_client = None

        # ── OpenAI-compatible providers ──────────────────────
        for p in OAI_PROVIDERS:
            key = _clean_env_value(os.getenv(p["env_key"]))
            if not key:
                continue

            try:
                client = OpenAI(api_key=key, base_url=p["base_url"])
                self.clients[p["name"]] = (client, p)
                self.fail_counts[p["name"]] = 0
                if p["name"] not in self.available:
                    self.available.append(p["name"])
            except Exception as e:
                print(f"⚠️ {p['name']} init failed: {e}")

        # ── OpenRouter ───────────────────────────────────────
        openrouter_key = _get_openrouter_key()
        if openrouter_key:
            self.clients["OpenRouter"] = (
                "openrouter_http",
                {
                    "name": "OpenRouter",
                    "model": OPENROUTER_MODELS[0],
                    "emoji": "",
                    "api_key": openrouter_key,
                    "base_url": "https://openrouter.ai/api/v1/chat/completions",
                    "referer": _clean_env_value(os.getenv("OPENROUTER_REFERER")) or "http://localhost",
                    "title": _clean_env_value(os.getenv("OPENROUTER_TITLE")) or "Algotec",
                },
            )
            self.fail_counts["OpenRouter"] = 0
            self.available.append("OpenRouter")

            if not _looks_like_openrouter_key(openrouter_key):
                preview = f"{openrouter_key[:8]}..." if openrouter_key else "(empty)"
                print(
                    f"⚠️ OpenRouter key found but format looks unusual: {preview} "
                    f"(expected to start with 'sk-or-')"
                )

        if claude_first and "Claude" in self.available:
            self.available.remove("Claude")
            self.available.insert(0, "Claude")

        if not self.available:
            raise EnvironmentError(
                "No AI provider keys found in .env\n"
                "Add at least one:\n"
                "  ANTHROPIC_API_KEY\n"
                "  DEEPSEEK_API_KEY\n"
                "  GROQ_API_KEY\n"
                "  GEMINI_KEY\n"
                "  OPENAI_KEY\n"
                "  OPENROUTER_API_KEY\n"
            )

        print(
            f"LLM Router ready [{'RBI mode' if claude_first else 'default'}] — "
            + " → ".join(self.available)
        )

    def _chat_claude(self, system_prompt: str, user_prompt: str, max_tokens: int) -> str:
        _, config = self.clients["Claude"]

        safe_max = min(max_tokens, 8192)
        combined_len = len(system_prompt) + len(user_prompt)
        if combined_len > 150_000:
            trim_to = max(1000, 150_000 - len(system_prompt) - 1000)
            user_prompt = user_prompt[:trim_to] + "\n[...truncated for length...]"

        response = self._anthropic_client.messages.create(
            model=config["model"],
            max_tokens=safe_max,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return response.content[0].text.strip()

    def _extract_openrouter_text(self, data: dict) -> str:
        choices = data.get("choices", [])
        if not choices:
            raise RuntimeError(f"OpenRouter returned no choices: {data}")

        message = choices[0].get("message", {})

        content = message.get("content")
        if isinstance(content, str) and content.strip():
            return content.strip()

        if isinstance(content, list):
            text_parts = []
            for item in content:
                if isinstance(item, dict):
                    if item.get("type") == "text" and item.get("text"):
                        text_parts.append(item["text"])
                    elif item.get("text"):
                        text_parts.append(item["text"])
            if text_parts:
                return "\n".join(text_parts).strip()

        reasoning = message.get("reasoning")
        if isinstance(reasoning, str) and reasoning.strip():
            return reasoning.strip()

        reasoning_details = message.get("reasoning_details")
        if isinstance(reasoning_details, list):
            text_parts = []
            for item in reasoning_details:
                if isinstance(item, dict) and item.get("text"):
                    text_parts.append(item["text"])
            if text_parts:
                return "\n".join(text_parts).strip()

        # Last-resort fallback for unusual providers
        raw_text = message.get("text")
        if isinstance(raw_text, str) and raw_text.strip():
            return raw_text.strip()

        raise RuntimeError(f"OpenRouter returned unsupported content format: {data}")

    def _chat_openrouter(self, system_prompt: str, user_prompt: str, max_tokens: int) -> str:
        _, config = self.clients["OpenRouter"]

        api_key = _clean_env_value(config["api_key"])
        if not api_key:
            raise RuntimeError("OpenRouter API key is empty after cleaning")

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": config["referer"],
            "X-OpenRouter-Title": config["title"],
        }

        last_err = None

        for idx, model_name in enumerate(OPENROUTER_MODELS):
            payload = {
                "model": model_name,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "max_tokens": max_tokens,
                "temperature": 0.1,
            }

            try:
                resp = requests.post(
                    config["base_url"],
                    headers=headers,
                    json=payload,
                    timeout=90,
                    allow_redirects=False,
                )
            except requests.RequestException as e:
                last_err = e
                continue

            if resp.status_code == 200:
                data = resp.json()
                if idx > 0:
                    print(f"↳ OpenRouter: using {model_name}")
                return self._extract_openrouter_text(data)

            err_text = resp.text
            err_lower = err_text.lower()
            last_err = RuntimeError(f"HTTP {resp.status_code}: {err_text}")

            if resp.status_code == 401:
                key_preview = f"{api_key[:8]}..." if api_key else "(empty)"
                raise RuntimeError(
                    "HTTP 401 from OpenRouter. Authorization header was sent, so the "
                    "API key is likely wrong/malformed/quoted. "
                    f"Key preview: {key_preview}. Response: {err_text}"
                )

            if resp.status_code == 404:
                # For explicit models, try the next one. If openrouter/free itself 404s,
                # surface it immediately because that's the primary path.
                if model_name == "openrouter/free":
                    raise RuntimeError(f"HTTP 404 from openrouter/free: {err_text}")
                print(f"↳ OpenRouter: {model_name} unavailable, trying next model...")
                continue

            if resp.status_code in {402, 408, 413, 429, 500, 502, 503, 504}:
                continue

            if "rate limit" in err_lower or "provider" in err_lower or "model" in err_lower:
                continue

            raise last_err

        raise last_err or RuntimeError("OpenRouter request failed")

    def _chat_oai(self, name: str, system_prompt: str, user_prompt: str, max_tokens: int) -> str:
        client, config = self.clients[name]

        groq_fallback_models = [
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
        ]

        models_to_try = groq_fallback_models if name == "Groq" else [config["model"]]
        last_err = None

        for model_name in models_to_try:
            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    max_tokens=max_tokens,
                    temperature=0.1,
                )
                if model_name != config["model"]:
                    print(f"↳ Groq: using {model_name} (fallback)")
                return response.choices[0].message.content.strip()

            except Exception as e:
                last_err = e
                err_s = str(e).lower()

                if name == "Groq" and (
                    "429" in err_s
                    or "rate" in err_s
                    or "model" in err_s
                    or "deprecat" in err_s
                    or "not found" in err_s
                ):
                    continue

                raise

        raise last_err

    def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 4096,
        prefer: str | None = None,
    ) -> str:
        order = list(self.available)

        if prefer and prefer in order:
            order.remove(prefer)
            order.insert(0, prefer)

        last_error = None

        for name in order:
            try:
                if name == "Claude":
                    result = self._chat_claude(system_prompt, user_prompt, max_tokens)
                elif name == "OpenRouter":
                    result = self._chat_openrouter(system_prompt, user_prompt, max_tokens)
                else:
                    result = self._chat_oai(name, system_prompt, user_prompt, max_tokens)

                self.fail_counts[name] = 0
                self.last_used = name

                if name != self.available[0]:
                    config = self.clients[name]
                    emoji = config[1]["emoji"] if isinstance(config, tuple) else "✨"
                    print(f"{emoji} [{name}] responded (fallback)")

                return result

            except Exception as e:
                err_str = str(e)
                err_lower = err_str.lower()
                err_code = getattr(e, "status_code", 0)

                self.fail_counts[name] = self.fail_counts.get(name, 0) + 1
                last_error = e

                should_fallback = (
                    err_code in FALLBACK_STATUS_CODES
                    or "insufficient" in err_lower
                    or "rate limit" in err_lower
                    or "quota" in err_lower
                    or "timeout" in err_lower
                    or "connection" in err_lower
                    or "overloaded" in err_lower
                    or "credit balance" in err_lower
                    or "too low" in err_lower
                    or "billing" in err_lower
                    or "not found" in err_lower
                    or "model" in err_lower
                    or "deprecated" in err_lower
                    or "401" in err_lower
                    or "404" in err_lower
                    or "400" in err_str
                    or "402" in err_str
                    or "413" in err_str
                    or "429" in err_str
                )

                if should_fallback:
                    try:
                        current_idx = order.index(name)
                        remaining = order[current_idx + 1 :]
                    except ValueError:
                        remaining = []

                    if remaining:
                        print(
                            f"⚠️ [{name}] failed ({type(e).__name__}: {err_str[:120]}) "
                            f"→ trying {remaining[0]}..."
                        )
                    else:
                        print(f"⚠️ [{name}] failed — all providers exhausted")
                    continue

                raise

        raise RuntimeError(
            f"All LLM providers failed.\n"
            f"Last error: {last_error}\n"
            f"Providers tried: {order}"
        )

    def status(self) -> dict:
        out = {}
        for name in list(self.clients.keys()):
            client_info = self.clients[name][1]
            out[name] = {
                "available": name in self.available,
                "fail_count": self.fail_counts.get(name, 0),
                "last_used": name == self.last_used,
                "model": client_info.get("model", "—"),
            }
        return out

    def print_status(self):
        print("\nLLM Router Status:")
        for name, info in self.status().items():
            if info["available"]:
                active = " ← active" if info["last_used"] else ""
                print(
                    f"✅ {name:<12} {info['model']:<35} "
                    f"fails:{info['fail_count']}{active}"
                )
            else:
                print(f"❌ {name:<12} (no API key)")
        print()


model = LLMRouter(claude_first=False)

try:
    rbi_model = LLMRouter(claude_first=True)
except Exception:
    rbi_model = model
    print("ℹ️ RBI model falling back to default (add ANTHROPIC_API_KEY for Claude)")