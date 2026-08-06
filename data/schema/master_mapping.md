# Schema — `master_mapping_canonical.csv`

The entity resolution table: every name observed anywhere in the source data mapped to a
canonical firm id, with its parent ecosystem.

**Real panel:** 193 canonical entities.
**Sample:** `data/samples/master_mapping_canonical.csv` — 5 fictional entities.

## Why this table decides whether the analysis works

Filings do not agree on names. "Amazon Web Services", "AWS" and "Amazon cloud
infrastructure" appear across different documents and mean one thing. If they stay three
nodes, every downstream measure breaks in the same direction and quietly understates the
answer:

- concentration looks like diversification — three providers instead of one
- `StructuralNonSubstitutability` reports alternatives that do not exist
- a shock scenario reaches a third of the firms it should
- the accumulation the whole project exists to detect disappears

This is unglamorous work and it is load-bearing. A perfect model on an unresolved graph
returns a confidently wrong answer.

## Columns

| Column | Meaning |
|---|---|
| `raw_node_id` | id of the raw observed string |
| `raw_name` | the name exactly as written in the source |
| `normalized_key` | lowercased/stripped form used for matching |
| `canonical_name` | the resolved display name |
| `canonical_firm_id` | **the node id used everywhere downstream** (`FIRM_nnnn`) |
| `entity_level` | `parent` / `subsidiary` / `affiliate` / `regional_entity` |
| `entity_type` | `company`, `cloud_provider`, `ai_lab`, `fintech`, `healthcare`, … |
| `parent_raw_name_hint` | parent as named in the source, if any |
| `parent_canonical_name` | resolved parent |
| `parent_firm_id` | **parent ecosystem id — used for concentration measures** |
| `is_parent_hint` | whether the row came from a parent reference |
| `source_files` | which panel the name was observed in |
| `observed_columns` | which column it appeared in |
| `years_observed` | years the name appears |
| `industries_observed` | industry labels seen alongside it |
| `raw_occurrence_count` | how often the raw string occurred |
| `mapping_status` | `mapped` / `unmapped` |
| `mapping_rule` | how it was resolved (exact, alias, parent hint, manual) |
| `confidence` | `high` / `medium` / `low` |
| `needs_review` | flagged for manual check |
| `notes` | free text |

## Entity level vs. parent ecosystem

Both matter, and for different reasons.

`canonical_firm_id` is the unit of *dependency*: a firm depending on Amazon Bedrock is not
the same edge as one depending on AWS EC2, and collapsing them would lose real structure.

`parent_firm_id` is the unit of *failure*. Concentration
(`dependency_concentration`) and substitutability (`structural_non_substitutability`) are
both computed over parent ecosystems, because AWS, Bedrock and SageMaker do not fail
independently. Counting them as three providers would be the same error as counting three
policies in one flood plain as geographic diversification.

`src/utils.parent_group()` resolves a node to its parent, falling back to itself for
independents.

## Resolution quality

`mapping_status`, `confidence` and `needs_review` exist because resolution is not fully
automatic and the failures are not random — obscure subsidiaries and non-English entity
names resolve worst, and those skew toward exactly the long-tail dependencies that make a
supply chain fragile. `src/step1_validate.py` reports every edge endpoint that fails to
resolve, since a missing mapping silently deletes a node from the graph rather than
throwing.
