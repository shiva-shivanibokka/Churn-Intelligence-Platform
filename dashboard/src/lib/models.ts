/**
 * The names of the models this dashboard actually runs.
 *
 * One place, because the sidebar and the agent route drifted apart: the footer
 * credited "Groq Qwen QwQ-32B" for months after the route had been switched to
 * Llama 3.3, so the page named a model no request ever went to. A label that
 * has to be updated by hand in a second file eventually is not.
 */

/**
 * Passed to Groq. Changing this changes what the sidebar credits.
 *
 * Hosted model names expire. This was `llama-3.3-70b-versatile` until Groq
 * retired it, at which point every agent request started returning
 * `404 model_not_found` — and because nothing exercised the live path, the
 * dashboard sat there for weeks with an agent that answered every question with
 * a 500. Neither CI nor a typecheck can catch it: the string is valid, the code
 * is correct, and the model simply stopped existing.
 *
 * If the agent starts failing, check this first — list what the key can reach:
 *     npx tsx -e "import G from 'groq-sdk'; new G().models.list().then(r => console.log(r.data.map(m => m.id).sort()))"
 */
export const AGENT_MODEL = "openai/gpt-oss-120b";

/** Human-readable form of the same thing, for UI. */
export const AGENT_MODEL_LABEL = "Groq GPT-OSS 120B";

/**
 * The 2D embedding shown on the segmentation page.
 *
 * It is PaCMAP, and has been since the reducer was swapped, but the function
 * was left named `fit_umap` and the label followed it everywhere — the page
 * heading announced a "UMAP 2D Projection" and the glossary defined Uniform
 * Manifold Approximation and Projection to people looking at a PaCMAP plot.
 * The database columns are still `umap_1` / `umap_2`; those are internal names
 * and renaming a live table buys nothing a reader sees.
 */
export const EMBEDDING_LABEL = "PaCMAP";
