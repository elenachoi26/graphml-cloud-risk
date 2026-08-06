# Risk Score Interpretation: 2022–2024

## Disclaimer

All scores are **weakly supervised graph-derived systemic exposure proxies**. They reflect structural dependency relationships extracted from 10-K filings and do **not** represent directly observed incident outcomes or financial losses.

## 1. Yearly Score Distribution

| Year | N | Mean | Median | Std | Min | Max | Skew |
|---|---|---|---|---|---|---|---|
| 2022 | 84 (11 source / 73 target) | 0.506 | 0.5673 | 0.2139 | 0.2363 | 0.8619 | -0.0731 |
| 2023 | 74 (11 source / 63 target) | 0.5068 | 0.5645 | 0.2095 | 0.2088 | 0.8554 | -0.0632 |
| 2024 | 72 (11 source / 61 target) | 0.5069 | 0.5306 | 0.2186 | 0.191 | 0.8764 | -0.137 |

## 2. Dominant Risk Component by Year

| Year | Dominant Component | Weighted Contribution | 2nd Component |
|---|---|---|---|
| 2022 | Direct Exposure | 0.2530 | Cascading Exposure (0.1012) |
| 2023 | Direct Exposure | 0.2534 | Cascading Exposure (0.1014) |
| 2024 | Direct Exposure | 0.2535 | Cascading Exposure (0.1014) |

## 3. Top 10 High-Risk Nodes by Year

### 2022

| Rank | Name | Type | CompositeRisk | Direct | Cascading | Concentration | NonSub |
|---|---|---|---|---|---|---|---|
| 1 | Red Hat | target_entity | 0.8619 | 0.988 | 0.940 | 0.696 | 0.405 |
| 2 | Cainiao | target_entity | 0.8298 | 1.000 | 0.750 | 0.696 | 0.405 |
| 3 | LinkedIn | target_entity | 0.8202 | 0.929 | 0.881 | 0.696 | 0.405 |
| 4 | VK Company | target_entity | 0.8107 | 0.976 | 0.714 | 0.696 | 0.405 |
| 5 | Nuance Communications | target_entity | 0.8095 | 0.917 | 0.857 | 0.696 | 0.405 |
| 6 | Kyndryl | target_entity | 0.8071 | 0.893 | 0.905 | 0.696 | 0.405 |
| 7 | GitHub | target_entity | 0.8012 | 0.905 | 0.845 | 0.696 | 0.405 |
| 8 | Lazada | target_entity | 0.7940 | 0.952 | 0.690 | 0.696 | 0.405 |
| 9 | JPMorgan Chase | target_entity | 0.7881 | 0.869 | 0.869 | 0.696 | 0.405 |
| 10 | Activision Blizzard | target_entity | 0.7869 | 0.881 | 0.833 | 0.696 | 0.405 |
### 2023

| Rank | Name | Type | CompositeRisk | Direct | Cascading | Concentration | NonSub |
|---|---|---|---|---|---|---|---|
| 1 | Nuance Communications | target_entity | 0.8554 | 0.986 | 0.892 | 0.709 | 0.419 |
| 2 | Red Hat | target_entity | 0.8473 | 0.959 | 0.919 | 0.709 | 0.419 |
| 3 | OpenAI | target_entity | 0.8459 | 0.973 | 0.878 | 0.709 | 0.419 |
| 4 | LinkedIn | target_entity | 0.8230 | 0.932 | 0.865 | 0.709 | 0.419 |
| 5 | GitHub | target_entity | 0.8135 | 0.919 | 0.851 | 0.709 | 0.419 |
| 6 | Activision Blizzard | target_entity | 0.8041 | 0.905 | 0.838 | 0.709 | 0.419 |
| 7 | ZeniMax Media | target_entity | 0.7811 | 0.865 | 0.824 | 0.709 | 0.419 |
| 8 | Whole Foods Market | target_entity | 0.7797 | 0.878 | 0.784 | 0.709 | 0.419 |
| 9 | Cerner | target_entity | 0.7784 | 1.000 | 0.473 | 0.709 | 0.419 |
| 10 | Rivian | target_entity | 0.7534 | 0.845 | 0.736 | 0.709 | 0.419 |
### 2024

| Rank | Name | Type | CompositeRisk | Direct | Cascading | Concentration | NonSub |
|---|---|---|---|---|---|---|---|
| 1 | Nuance Communications | target_entity | 0.8764 | 1.000 | 0.958 | 0.701 | 0.444 |
| 2 | OpenAI | target_entity | 0.8667 | 0.986 | 0.944 | 0.701 | 0.444 |
| 3 | LinkedIn | target_entity | 0.8542 | 0.972 | 0.917 | 0.701 | 0.444 |
| 4 | GitHub | target_entity | 0.8306 | 0.931 | 0.903 | 0.701 | 0.444 |
| 5 | Red Hat | target_entity | 0.8306 | 0.958 | 0.833 | 0.701 | 0.444 |
| 6 | HashiCorp | target_entity | 0.8042 | 0.917 | 0.806 | 0.701 | 0.444 |
| 7 | Activision Blizzard | target_entity | 0.7903 | 0.861 | 0.875 | 0.701 | 0.444 |
| 8 | Siemens Digital Industries Software | target_entity | 0.7861 | 0.903 | 0.750 | 0.701 | 0.444 |
| 9 | Whole Foods Market | target_entity | 0.7722 | 0.847 | 0.819 | 0.701 | 0.444 |
| 10 | TikTok | target_entity | 0.7486 | 0.944 | 0.458 | 0.701 | 0.444 |

## 4. Year-over-Year Risk Changes

### 4a. Consistently High-Risk Nodes (mean > 0.65, std < 0.10)

| Name | Type | 2022 | 2023 | 2024 | Mean | Std |
|---|---|---|---|---|---|---|
| Nuance Communications | target_entity | 0.8095 | 0.8554 | 0.8764 | 0.8471 | 0.0279 |
| Red Hat | target_entity | 0.8619 | 0.8473 | 0.8306 | 0.8466 | 0.0128 |
| LinkedIn | target_entity | 0.8202 | 0.8230 | 0.8542 | 0.8325 | 0.0154 |
| GitHub | target_entity | 0.8012 | 0.8135 | 0.8306 | 0.8151 | 0.0120 |
| Activision Blizzard | target_entity | 0.7869 | 0.8041 | 0.7903 | 0.7937 | 0.0074 |
| Whole Foods Market | target_entity | 0.7702 | 0.7797 | 0.7722 | 0.7741 | 0.0041 |
| Rivian | target_entity | 0.7619 | 0.7534 | 0.7479 | 0.7544 | 0.0058 |
| Cerner | target_entity | 0.7476 | 0.7784 | 0.7083 | 0.7448 | 0.0287 |
| iRobot | target_entity | 0.7131 | 0.7534 | 0.7479 | 0.7381 | 0.0178 |
| Cainiao | target_entity | 0.8298 | 0.7027 | 0.6417 | 0.7247 | 0.0784 |
| Lazada | target_entity | 0.7940 | 0.6757 | 0.6819 | 0.7172 | 0.0544 |
| Alibaba Cloud | target_entity | 0.6601 | 0.7264 | 0.6806 | 0.6890 | 0.0277 |
| Sun Art Retail | target_entity | 0.6970 | 0.5946 | 0.7125 | 0.6680 | 0.0523 |
| One Medical | target_entity | 0.6315 | 0.6682 | 0.6882 | 0.6627 | 0.0235 |
| MGM Holdings | target_entity | 0.6315 | 0.6682 | 0.6882 | 0.6627 | 0.0235 |

### 4b. Volatile Nodes (std > 0.12)

| Name | Type | 2022 | 2023 | 2024 | Std | Δ22→24 |
|---|---|---|---|---|---|---|
| Amazon Web Services | target_entity | 0.2363 | 0.7297 | 0.2188 | 0.2368 | -0.0176 |
| Altair Engineering | source_firm | 0.6619 | 0.2473 | 0.5458 | 0.1746 | -0.1161 |
| IBM | source_firm | 0.6113 | 0.4541 | 0.2771 | 0.1365 | -0.3342 |

## 5. Interpretation Notes

- **Score construction**: CompositeRisk = 0.50 × DirectExposure_norm + 0.20 × CascadingExposure_norm + 0.20 × DependencyConcentration_norm + 0.10 × NonSubstitutability_norm. Each component is year-wise percentile rank normalized before combining.
- **Target entity dominance**: Most high-scoring nodes are `target_entity` nodes (subsidiaries/affiliates of source firms). Their high concentration scores reflect single-parent dependency (HHI = 1.0 when all upstream dependency flows from one parent group).
- **Source firm interpretation**: High-scoring source firms indicate they are exposed downstream (many entities depend on them) AND have structural concentration in their upstream supply chain.
- **DirectExposure is typically the dominant driver** across all three years, reflecting that direct upstream dependency relationships carry the most structural risk weight.
- **CascadingExposure is low on average** (mean normalized ~0.02–0.03 raw), indicating the current dataset mainly captures 1-hop dependency relationships. This is expected for 10-K disclosure data.
- **YoY stability**: nodes that consistently score high are structurally embedded — high PageRank as upstream hubs or concentrated single-parent dependency.
- **These scores do not imply any firm will fail or experience an adverse event.**