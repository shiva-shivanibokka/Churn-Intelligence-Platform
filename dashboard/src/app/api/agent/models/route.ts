import { NextRequest, NextResponse } from "next/server";
import OpenAI from "openai";
import { resolveProvider, scrubKeys } from "@/lib/providers";

/**
 * List the models a visitor's key can actually reach.
 *
 * This exists because a hardcoded model name is a time bomb. The agent was
 * pinned to `llama-3.3-70b-versatile` until Groq retired it, after which every
 * request returned `404 model_not_found` and surfaced in the browser as a
 * generic 500 — for weeks, because no test exercises the live LLM path and the
 * name is a valid string in correct code that stopped existing. Nothing in CI
 * can catch that. The provider is the only authority on what it serves, so the
 * dropdown asks it.
 *
 * The key arrives in a header, is used for this one request, and is never
 * stored or logged. The base URL comes from the fixed provider map, never from
 * the request — a caller-supplied endpoint alongside a caller-supplied key
 * would make this an open proxy.
 */
export const dynamic = "force-dynamic";

/** Models that cannot hold a tool-calling conversation, whatever they are named. */
const NOT_CHAT = /whisper|tts|embed|guard|moderation|orpheus|distil|safeguard|rerank|vision-only/i;

export async function POST(req: NextRequest) {
  let provider;
  try {
    const body = await req.json();
    provider = resolveProvider(body?.provider);
  } catch (err) {
    return NextResponse.json({ error: String(err) }, { status: 400 });
  }

  const key = req.headers.get("x-llm-key")?.trim();
  if (!key) {
    return NextResponse.json(
      { error: `Add a ${provider.label} API key to list its models.`, needs_key: true },
      { status: 401 }
    );
  }

  const client = new OpenAI({
    apiKey: key,
    baseURL: provider.baseUrl,
    timeout: 15_000,
    maxRetries: 1,
  });

  try {
    const list = await client.models.list();
    const models = list.data
      .map((m) => m.id)
      .filter((id) => !NOT_CHAT.test(id))
      .sort();

    // Some providers do not expose /models, or expose it without the chat
    // models in it. An empty list is not an error — it means "we could not
    // enumerate", and the UI should let the visitor type a name instead of
    // presenting an empty dropdown as if nothing were available.
    return NextResponse.json({
      models,
      // Only suggest the configured default if the provider actually lists it.
      suggested: models.includes(provider.defaultModel)
        ? provider.defaultModel
        : models[0] ?? provider.defaultModel,
      enumerated: models.length > 0,
    });
  } catch (err: unknown) {
    const status = (err as { status?: number })?.status;
    if (status === 401 || status === 403) {
      return NextResponse.json(
        { error: `${provider.label} rejected that key.`, invalid_key: true },
        { status: 401 }
      );
    }
    // A provider without a /models endpoint is usable, just not browsable.
    console.error(`models list failed for ${provider.id}:`, scrubKeys(String(err)));
    return NextResponse.json({
      models: [],
      suggested: provider.defaultModel,
      enumerated: false,
      error: `${provider.label} did not return a model list, so type a model name instead.`,
    });
  }
}
