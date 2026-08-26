/**
 * The names of the models this dashboard actually runs.
 *
 * One place, because the sidebar and the agent route drifted apart: the footer
 * credited "Groq Qwen QwQ-32B" for months after the route had been switched to
 * Llama 3.3, so the page named a model no request ever went to. A label that
 * has to be updated by hand in a second file eventually is not.
 */

/** Passed to Groq. Changing this changes what the sidebar credits. */
export const AGENT_MODEL = "llama-3.3-70b-versatile";

/** Human-readable form of the same thing, for UI. */
export const AGENT_MODEL_LABEL = "Groq Llama 3.3 70B";

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
