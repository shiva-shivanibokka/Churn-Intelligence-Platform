/**
 * The LLM providers the agent can run against.
 *
 * All of them speak the OpenAI chat-completions dialect, including tool
 * calling, so the agent has one code path and the provider is a base URL plus a
 * credential. That is why the `openai` SDK is used rather than each vendor's
 * own — `groq-sdk` worked fine when Groq was the only option, but a second SDK
 * per provider means a second tool-calling shape, a second error taxonomy and a
 * second retry policy to keep in step.
 *
 * ── Why this list is a fixed map ──────────────────────────────────────────────
 *
 * The base URL is chosen here, by key, and never taken from the request. The
 * agent route accepts a caller-supplied API key, and a caller-supplied *endpoint*
 * alongside it would turn the route into an open proxy: anything with a `fetch`
 * could point it at an internal address and read the response back through us.
 * The client sends a provider id; the server resolves the URL.
 *
 * ── Why models are fetched, not listed ────────────────────────────────────────
 *
 * There is no hardcoded model catalogue here, on purpose. This project shipped a
 * dead agent for weeks because Groq retired `llama-3.3-70b-versatile` and every
 * request began returning `404 model_not_found` — a valid string in correct code
 * that simply stopped existing, which no test or type check can catch. Model
 * names are the provider's to know, so `/api/agent/models` asks them with the
 * visitor's key and the UI offers what comes back.
 *
 * `defaultModel` is only a starting selection, and it is allowed to be wrong:
 * if it is not in the fetched list, the UI falls back to the first model the
 * provider actually offers.
 */

export type ProviderId = "groq" | "openai" | "anthropic" | "gemini" | "openrouter" | "cerebras";

export type Provider = {
  id: ProviderId;
  label: string;
  /** OpenAI-compatible base URL. Resolved server-side, never sent by the client. */
  baseUrl: string;
  /** Where a visitor gets a key. */
  keyUrl: string;
  /** Shape check for a pasted key — a typo filter, not authentication. */
  keyPattern: RegExp;
  /** Human hint for the key format. */
  keyHint: string;
  /** Starting selection. Superseded by whatever the provider actually lists. */
  defaultModel: string;
  /** Whether the free tier is generous enough to matter to someone trying this. */
  note: string;
};

export const PROVIDERS: Record<ProviderId, Provider> = {
  groq: {
    id: "groq",
    label: "Groq",
    baseUrl: "https://api.groq.com/openai/v1",
    keyUrl: "https://console.groq.com/keys",
    keyPattern: /^gsk_[A-Za-z0-9]{20,}$/,
    keyHint: "gsk_…",
    defaultModel: "openai/gpt-oss-120b",
    note: "Free tier, no card. The quickest way to try this.",
  },
  openai: {
    id: "openai",
    label: "OpenAI",
    baseUrl: "https://api.openai.com/v1",
    keyUrl: "https://platform.openai.com/api-keys",
    keyPattern: /^sk-[A-Za-z0-9_-]{20,}$/,
    keyHint: "sk-…",
    defaultModel: "gpt-4o-mini",
    note: "Paid. Pennies for a handful of agent runs.",
  },
  anthropic: {
    id: "anthropic",
    label: "Anthropic",
    // Anthropic exposes an OpenAI-compatible surface at /v1 alongside its own API.
    baseUrl: "https://api.anthropic.com/v1",
    keyUrl: "https://console.anthropic.com/settings/keys",
    keyPattern: /^sk-ant-[A-Za-z0-9_-]{20,}$/,
    keyHint: "sk-ant-…",
    defaultModel: "claude-sonnet-4-5",
    note: "Paid. Strong tool calling.",
  },
  gemini: {
    id: "gemini",
    label: "Google Gemini",
    baseUrl: "https://generativelanguage.googleapis.com/v1beta/openai",
    keyUrl: "https://aistudio.google.com/apikey",
    keyPattern: /^[A-Za-z0-9_-]{30,}$/,
    keyHint: "AIza…",
    defaultModel: "gemini-2.0-flash",
    note: "Free tier available.",
  },
  openrouter: {
    id: "openrouter",
    label: "OpenRouter",
    baseUrl: "https://openrouter.ai/api/v1",
    keyUrl: "https://openrouter.ai/keys",
    keyPattern: /^sk-or-[A-Za-z0-9_-]{20,}$/,
    keyHint: "sk-or-…",
    defaultModel: "openai/gpt-4o-mini",
    note: "One key, many models. Some are free.",
  },
  cerebras: {
    id: "cerebras",
    label: "Cerebras",
    baseUrl: "https://api.cerebras.ai/v1",
    keyUrl: "https://cloud.cerebras.ai",
    keyPattern: /^csk-[A-Za-z0-9]{20,}$/,
    keyHint: "csk-…",
    defaultModel: "llama-3.3-70b",
    note: "Free tier. Very fast inference.",
  },
};

export const PROVIDER_IDS = Object.keys(PROVIDERS) as ProviderId[];

/** The default a first-time visitor lands on. */
export const DEFAULT_PROVIDER: ProviderId = "groq";

export function isProviderId(value: unknown): value is ProviderId {
  return typeof value === "string" && value in PROVIDERS;
}

/**
 * Resolve a provider id to its configuration, server-side.
 *
 * Throws on anything unrecognised rather than falling back to a default: an
 * unknown provider id means the request did not come from this UI, and quietly
 * substituting one would send the visitor's credential somewhere they did not
 * choose.
 */
export function resolveProvider(id: unknown): Provider {
  if (!isProviderId(id)) {
    throw new Error(
      `Unknown provider ${JSON.stringify(id)}. Expected one of: ${PROVIDER_IDS.join(", ")}.`
    );
  }
  return PROVIDERS[id];
}

/**
 * Redact anything key-shaped from text on its way to a log or a browser.
 *
 * One pattern per provider, plus a catch-all for long `sk-`-prefixed strings,
 * because provider SDKs sometimes echo request context into error messages and
 * this route returns those messages to the caller.
 */
export function scrubKeys(text: string): string {
  return text
    .replace(/gsk_[A-Za-z0-9]{20,}/g, "gsk_***")
    .replace(/csk-[A-Za-z0-9]{20,}/g, "csk-***")
    .replace(/sk-ant-[A-Za-z0-9_-]{20,}/g, "sk-ant-***")
    .replace(/sk-or-[A-Za-z0-9_-]{20,}/g, "sk-or-***")
    .replace(/sk-[A-Za-z0-9_-]{20,}/g, "sk-***")
    .replace(/AIza[A-Za-z0-9_-]{20,}/g, "AIza***");
}
