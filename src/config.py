"""
Central configuration — every path, year and scoring weight lives here.

Nothing else in the codebase hard-codes a weight or a directory. If a number in
`docs/03-risk-formulas.md` disagrees with a number here, this file is wrong.
"""

from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
# Everything resolves from the repository root, so the pipeline runs from any cwd.
ROOT      = Path(__file__).resolve().parent.parent
DATA_DIR  = ROOT / "data"
RAW_DIR   = DATA_DIR / "raw"        # git-ignored; see data/README.md
SAMPLE_DIR = DATA_DIR / "samples"   # synthetic stand-ins, committed
FIG_DIR   = ROOT / "figures"

# Input tables. Resolved against RAW_DIR first, then SAMPLE_DIR (see utils.load_input).
INPUT_FILES = {
    "edges":    "edges_risk_propagation_valid.csv",
    "self":     "self_features_mapped.csv",
    "incident": "incidents_mapped.csv",
    "master":   "master_mapping_canonical.csv",
}

# Read location vs. write location — they are not always the same.
#
# `outputs/` holds the real committed results and is what everything *reads*.
# They cannot be regenerated: the raw panels are not distributed. So if a clean
# clone ran the pipeline against the synthetic samples and wrote to `outputs/`,
# it would replace 82 real nodes with 5 fictional ones, permanently.
#
# Hence: a sample run writes to `outputs_sample/` (git-ignored) instead. The
# model and reporting stages still read `outputs/`, so the committed results
# stay inspectable and reproducible-from even on a clone with no raw data.
USING_SAMPLES = not (RAW_DIR / INPUT_FILES["edges"]).exists()

OUT_DIR   = ROOT / "outputs"                                  # canonical, read
WRITE_DIR = ROOT / "outputs_sample" if USING_SAMPLES else OUT_DIR

MODEL_DIR    = OUT_DIR / "models"
PRED_DIR     = OUT_DIR / "prediction_2025"
SCENARIO_DIR = OUT_DIR / "scenario_propagation"
REPORT_DIR   = OUT_DIR / "interpretation_report"

# ── Panel ─────────────────────────────────────────────────────────────────────
YEARS       = [2022, 2023, 2024, 2025]
TRAIN_YEARS = [2022, 2023, 2024]   # weak-label years
PREDICT_YEAR = 2025                # held out; scored by the trained GCN

# Relations that carry no risk-propagation semantics. A competitor's outage does
# not interrupt your service, so these edges are dropped before graph assembly.
REMOVE_RELATIONS = {"competitor", "regulator", "investor", "investee"}

# ── Step 3 — Edge weight ──────────────────────────────────────────────────────
# edge_weight = max(FLOOR, w_css·CSS + w_rls·RLS + w_noalt·NoAlt + w_inv·Inv)
EW_CSS   = 0.40   # contract_scale_score, min-max normalised
EW_RLS   = 0.30   # risk_language_score, min-max normalised
EW_NOALT = 0.20   # "sole source" / no-alternative flag
EW_INV   = 0.10   # strategic investment or long-term partnership flag
EW_FLOOR = 0.10   # a disclosed relationship is never weightless

# ── Step 4 — Node vulnerability ───────────────────────────────────────────────
# network_criticality = NC_PR·pagerank + NC_IDC·indegree_centrality
# node_vulnerability  = NV_IR·intrinsic + NV_INC·incident + NV_NC·network_criticality
NC_PR  = 0.50
NC_IDC = 0.50
NV_IR  = 0.50
NV_INC = 0.25
NV_NC  = 0.25

# The six 10-K risk features averaged into IntrinsicRisk. Only source firms have
# these — target entities get 0.0 and are carried by structure alone.
INTRINSIC_FEATURES = [
    "cybersecurity_risk",
    "operational_continuity_risk",
    "ai_dependency_risk",
    "regulatory_risk",
    "geopolitical_risk",
    "energy_infra_risk",
]

# ── Step 6 — Composite risk ───────────────────────────────────────────────────
# CompositeRisk = 0.50·Direct + 0.20·Cascading + 0.20·Concentration + 0.10·NonSub
# Direct exposure dominates because it is the component an underwriter can act on
# at renewal; the structural terms modulate it.
CR_DIRECT  = 0.50
CR_CASCADE = 0.20
CR_CONC    = 0.20
CR_NONSUB  = 0.10

COMPOSITE_WEIGHTS = {
    "direct_exposure":                 CR_DIRECT,
    "cascading_exposure":              CR_CASCADE,
    "dependency_concentration":        CR_CONC,
    "structural_non_substitutability": CR_NONSUB,
}

assert abs(sum(COMPOSITE_WEIGHTS.values()) - 1.0) < 1e-9, "CompositeRisk weights must sum to 1"

# ── Stage 3 — GNN ─────────────────────────────────────────────────────────────
# Graph-derived quantities (pagerank, indegree, node_vulnerability, the exposure
# components) are deliberately kept out of the feature set: the model must learn
# structure from the graph, not read it off an input column. Otherwise the
# tabular baseline is not a fair control.
GNN_FEATURES = INTRINSIC_FEATURES + ["incident_signal", "node_type_enc"]
NODE_TYPE_MAP = {"source_firm": 0, "target_entity": 1}

CV_FOLDS  = 5
GNN_HIDDEN = 32
GNN_EPOCHS = 100
GNN_LR     = 0.005
RANDOM_SEED = 42

# ── Stage 4 — Scenario propagation ────────────────────────────────────────────
# path_score = severity × ∏ edge_weight along the path × DECAY^(hop-1)
#
# Edge weights already attenuate the signal multiplicatively; DECAY is an
# additional per-hop penalty for the fact that distance itself buys time —
# a firm three hops from an outage learns about it later and has more room to
# fail over than one wired directly into it.
SCENARIO_MAX_HOPS = 3
SCENARIO_DECAY    = 0.85
SCENARIO_DEFAULT_SEVERITY = 1.0

# Columns holding verbatim 10-K sentences. Stripped from anything published —
# see tools/scrub_outputs.py and data/README.md.
VERBATIM_TEXT_COLUMNS = [
    "근거 문장",
    "evidence_sentence",
    "last_evidence_sentence",
    "edge_chain",
]


def ensure_dirs() -> None:
    """Create every output directory the pipeline writes into."""
    for d in (OUT_DIR, WRITE_DIR, MODEL_DIR, PRED_DIR, SCENARIO_DIR, REPORT_DIR):
        d.mkdir(parents=True, exist_ok=True)
