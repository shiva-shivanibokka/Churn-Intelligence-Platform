-- ─── retention_playbook ───────────────────────────────────────────────────────
-- Drives the search_retention_playbook AI agent tool.
-- Edit rows here instead of touching source code.

CREATE TABLE IF NOT EXISTS retention_playbook (
  id                   UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  risk_factor_keyword  TEXT NOT NULL UNIQUE,
  intervention         TEXT NOT NULL,
  message              TEXT NOT NULL,
  cost                 TEXT NOT NULL
);

INSERT INTO retention_playbook (risk_factor_keyword, intervention, message, cost) VALUES
  ('satisfaction',          'Proactive Support Call', 'Reach out to understand and resolve the satisfaction issue before they escalate.',               'Medium ($10–25)'),
  ('complain',              'Service Recovery',       'Apologise directly, offer a goodwill gesture, and close the loop on the complaint.',             'Medium ($10–25)'),
  ('days_since_last_order', 'Reactivation Campaign',  'Send a personalised win-back offer highlighting new products relevant to past purchases.',        'Low ($1–5)'),
  ('tenure',                'Loyalty Reward',         'Recognise their loyalty with an exclusive member reward to reinforce the relationship.',          'Low ($1–5)'),
  ('order_count',           'Personalized Content',   'Send curated product recommendations based on their purchase history to re-engage browsing.',     'Low ($1–5)'),
  ('cashback',              'Discount Offer',         'Offer a targeted cashback or discount on their next order to incentivise return.',                'Low ($1–5)'),
  ('default',               'Personalized Content',   'Send a personalised re-engagement email with relevant content.',                                  'Low ($1–5)')
ON CONFLICT (risk_factor_keyword) DO NOTHING;


-- ─── business_config ──────────────────────────────────────────────────────────
-- Key-value store for business parameters.
-- Change values here; the AI agent reads them at runtime — no code deploy needed.

CREATE TABLE IF NOT EXISTS business_config (
  key         TEXT PRIMARY KEY,
  value       TEXT NOT NULL,
  description TEXT
);

INSERT INTO business_config (key, value, description) VALUES
  (
    'assumed_clv_usd',
    '500',
    'Assumed average Customer Lifetime Value in USD. Used for revenue-at-risk calculations. Update when you have real CLV data.'
  ),
  (
    'intervention_types',
    'Discount Offer,Loyalty Reward,Personalized Content,Proactive Support Call,Reactivation Campaign,Premium Upgrade,Service Recovery',
    'Comma-separated list of valid intervention types. The AI agent will only generate types from this list.'
  ),
  (
    'channels',
    'In-App Notification,Email,Push Notification,Direct Call,SMS',
    'Comma-separated list of valid outreach channels.'
  ),
  (
    'timing_options',
    'Immediate,Within 24 hours,Within 1 week,During next session',
    'Comma-separated list of valid timing options.'
  )
ON CONFLICT (key) DO NOTHING;
