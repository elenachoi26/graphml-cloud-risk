# Data Quality Report

## Edge filtering
- Removed 124 edges with relations: {'competitor', 'regulator', 'investee', 'investor'}
- Remaining edges: 446

## Canonical ID coverage
- Edge node IDs not in master mapping: 0

## Duplicate edges
- Duplicate edge rows: 4

## Missing values in edge weight columns
- contract_scale_score: 0 missing
- risk_language_score: 0 missing
- no_alternative_mentioned: 0 missing
- investment_mentioned: 0 missing

## Self-feature (source_firm) coverage
- Firms with 10-K self features: 12
  - 2022: 12 firms
  - 2023: 12 firms
  - 2024: 12 firms
  - 2025: 12 firms

## Incident coverage
- Firms with incident records: 37

## IntrinsicRisk features found in self_features
- ['cybersecurity_risk', 'operational_continuity_risk', 'ai_dependency_risk', 'regulatory_risk', 'geopolitical_risk', 'energy_infra_risk']
