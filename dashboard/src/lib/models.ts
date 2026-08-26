/**
 * The names of the models this dashboard actually runs.
 *
 * One place, because the sidebar and the agent route drifted apart: the footer
 * credited "Groq Qwen QwQ-32B" for months after the route had been switched to
 * Llama 3.3, so the page named a model no request ever went to. A label that
 * has to be updated by hand in a second file eventually is not.
 */

/**
 * What the sidebar credits when no visitor has chosen anything.
 *
 * The agent model used to be a build-time constant here, which is how this
 * project shipped a dead agent for weeks: Groq retired the name it held, every
 * request returned `404 model_not_found`, and nothing in a type check or test
 * suite can notice a valid string that stopped existing.
 *
 * It is no longer a constant. Each visitor picks a provider and a model from a
 * list fetched off that provider with their own key — see `src/lib/providers.ts`
 * — so there is nothing here to go stale. This label is only what the footer
 * says before anyone has chosen.
 */
export const AGENT_MODEL_LABEL = "Bring your own model";

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
