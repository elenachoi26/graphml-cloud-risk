# 1. The problem

## Concentration turns independent policies into one bet

AI-enabled services run on infrastructure that very few companies supply. As of 2024 the
cloud IaaS market was majority-held by three providers — AWS (31%), Microsoft Azure (24%),
Google Cloud (11%) — and the foundation-model layer is similarly concentrated among OpenAI,
Anthropic, Google and Meta. Those two layers are not independent either: most foundation
models are strategically tied to a specific cloud, so dependence stacks rather than spreads.

The July 2024 CrowdStrike outage is the shape of the resulting loss. One faulty update
affected roughly 8.5 million Windows devices, generated an estimated USD 5.4 billion of
loss across Fortune 500 firms, and an estimated USD 540 million to 1.5 billion of insured
cyber loss. No attacker, no breach — a single software update.

## This is catastrophe accumulation, relocated

Property insurers have solved a version of this. In natural catastrophe insurance, writing
too many policies in one flood plain means one event triggers every claim at once, so cat
underwriters track exposure by geography and cap it.

AI infrastructure risk has the same structure with a different accumulation unit. It is not
geography — it is the cloud provider, the foundation model, the API, the SaaS platform. An
insurer can hold a portfolio that looks beautifully diversified across industry, size and
region, and still have most of it sitting downstream of one cloud region.

The difference is that a flood plain is visible on a map, and this is not. That is the gap
this project addresses.

## Where it bites in the insurance value chain

**Underwriting.** Contingent business interruption (CBI) covers losses caused by a
third-party provider's failure; Technology E&O covers failure to deliver a promised service
when an upstream API goes down. Both require quantifying externally-originated risk
transmission. Existing questionnaires do not ask about it. And as AI moves into autonomous
vehicles, medical diagnosis, smart factories and financial operations, the same exposure
reaches auto, medical liability, property and operational-risk lines.

**Portfolio / CRO.** This is the material one. Insured firms can be diversified across
industries and regions and still share a cloud region or a model API. At the contract level
each failure looks independent. At the portfolio level, common provider dependence produces
correlated loss — a tail event assembled from policies that were each priced as if
independent.

**Regulation.** BIS and FSB have both flagged AI supply-chain concentration and third-party
dependence as emerging systemic vulnerabilities. That literature is largely descriptive: it
establishes that the risk exists without offering a way to measure structural exposure at
the portfolio level. A measurement framework is the missing piece.

## Why existing underwriting cannot see it

### It asks the insured about itself

Cyber underwriting evolved around internal security maturity: MFA, patch management,
backups, anti-malware, remote access controls, incident response. That works well for
ransomware, breaches and intrusions.

It does not work here, for two reasons. First, **the loss needs no security incident at
all** — a cloud region outage, a model API interruption, an auth service error or a platform
policy change all produce business interruption with the insured's controls fully intact.
Second, **the assessment unit is one firm at a time.** Even a perfect per-firm evaluation
cannot reveal that forty of your insureds share an upstream provider, because that fact does
not exist in any single application form.

### The dependency data is not structured

The information is genuinely public — 10-K filings, partnership announcements, product
documentation, press releases all disclose cloud usage, platform dependence and strategic
partnerships. But it is scattered through unstructured prose. Nothing organises it into a
dataset that can answer, consistently and at scale: who depends on whom, is that dependency
core or incidental, and does an alternative exist?

### The math points at a term nobody measures

Portfolio loss is `L = Σ wᵢLᵢ`, and its variance decomposes:

```
Var(L_portfolio) = Σ wᵢ² Var(Lᵢ)  +  Σ(i≠j) wᵢwⱼ Cov(Lᵢ, Lⱼ)
                   └── individual ──┘   └──── correlated ────┘
```

Traditional underwriting prices the first term well — that is what per-firm risk assessment
produces. AI supply-chain risk lives almost entirely in the second. Every additional insured
firm depending on the same cloud region raises the covariance term, and nothing in a
per-firm questionnaire will ever surface it.

Measuring covariance requires knowing which firms share which dependencies. That is a
network question, and a table of firm attributes cannot answer it — no matter how good the
model on top of the table is. → [`04-model.md`](04-model.md) tests exactly that claim, and
it holds: the tabular baseline does not merely underperform, it ranks exposure backwards.

---

Next: [2. Methodology](02-methodology.md) — how the graph is built.
