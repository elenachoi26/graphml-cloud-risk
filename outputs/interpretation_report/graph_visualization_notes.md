# Graph Visualization Notes

## Visual Encoding

- **Node size**: proportional to CompositeRisk (larger = higher risk score)
- **Node color**: red = `source_firm` (focal 10-K filer); blue = `target_entity` (external provider/service)
- **Edge width**: proportional to `edge_weight` (0.40×contract_scale + 0.30×risk_language + 0.20×no_alt + 0.10×investment)
- **Edge direction**: upstream provider → downstream exposed firm (risk propagation direction)
- **Red border**: top-10 highest composite risk nodes
- Labels shown for: all source_firm nodes, target_entity nodes with composite_risk ≥ 0.55, all top-risk highlighted nodes

## Subgraph Composition

- Top-10 high-risk nodes (red)
- Their direct upstream predecessors (orange, 1-hop)
- 2-hop upstream nodes (light blue)

## Ego Network Interpretation

- Focal firm is shown in red (center of the ego network)
- Orange nodes are upstream providers (in-edges toward focal firm)
- Blue nodes are downstream dependents (out-edges from focal firm)
- Ego graphs show 1-hop neighborhood only (radius=1)

## Layout

- Spring layout (Fruchterman-Reingold) with fixed seed for reproducibility
- Node repulsion `k = 2/sqrt(N)` to reduce overlap

## Coverage

- **2022**: 84 nodes, 111 edges — most central node by PageRank: **SAP**
- **2023**: 74 nodes, 96 edges — most central node by PageRank: **SAP**
- **2024**: 72 nodes, 96 edges — most central node by PageRank: **SAP**
- **2025**: 82 nodes, 138 edges — most central node by PageRank: **MongoDB**