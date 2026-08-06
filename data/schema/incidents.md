# Schema — `incidents_mapped.csv`

External risk events matched to graph entities. Supplies the `IncidentSignal` term in node
vulnerability: what has actually happened to a node, as opposed to what it says about itself
(`self_features`) or where it sits (`network_criticality`).

> **The real panel came from a commercial risk-data provider under licence and is not
> redistributed here — neither the records nor anything derived from them.** The pipeline
> reads any table matching this schema; `data/samples/incidents_mapped.csv` is a synthetic
> stand-in. Public alternatives with comparable structure exist (CVE/CISA advisories, cloud
> status-page histories, breach notification registries), and the column contract below is
> what the pipeline actually depends on.

## Columns the pipeline uses

| Column | Meaning |
|---|---|
| `company_canonical_firm_id` | resolved node id — the join key |
| `severity` | how damaging the event was |
| `reach` | how widely it was felt |
| `novelty` | how unprecedented it was |

That is the whole contract. Everything else is provenance.

## Other columns

`incident_id`, `incident_date`, `company_name`, `headquarters_country_isocode`,
`related_countries`, topical flags (`ai`, `cyberattack`, `energy_management`),
and the resolution trail (`company_canonical_name`, `company_entity_level`,
`company_entity_type`, `company_parent_*`, `company_mapping_found`).

## How the signal is computed

```
incident_score_raw(firm) = Σ over events of (severity × reach × novelty)
incident_signal(firm)    = minmax(incident_score_raw)      # → [0, 1]
```

The three factors **multiply** rather than sum: a severe incident nobody noticed and a
trivial one that made global news are both less informative than an event that is severe,
wide-reaching and unprecedented. Summing would let a long tail of minor events outweigh one
genuine crisis.

## Known limitation

Events are pooled per firm across all years — the signal is static, not time-varying. With
1,071 events over 193 entities, slicing by year would leave most firm-years empty and make
the term mostly noise. The honest reading is that `IncidentSignal` describes a firm's overall
incident history, not its state in a given year, and at 0.25 weight it is a prior on
node fragility rather than a temporal signal.

Implemented in `src/step4_node_vulnerability.py`.
