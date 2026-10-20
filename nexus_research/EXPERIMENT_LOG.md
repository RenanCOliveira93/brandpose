# Experiment log — Attention Intelligence

## Baseline

- Date: 2026-10-19
- Original upstream: `escudero/brandpose`
- Fork: `RenanCOliveira93/brandpose`
- Fork baseline commit: `9e9665c2601409f3b23da82256e662dede07f5d8`
- Experimental branch: `nexus/research-attention-v0`
- Isolation: no changes to `retail-solutions`; no integration or database writes.
- v0.0 implementation: offline zone aggregation and synthetic tests only.
- Runtime/model benchmarks: **NOT RUN**.
- Real video validation: **NOT RUN**.
- Human annotated ground truth: **NOT AVAILABLE**.
- License/commercial readiness: **NOT APPROVED**.

## Experiment template

| Field | Value |
|---|---|
| Experiment ID | TBD |
| Commit SHA | TBD |
| Environment, OS, CPU/GPU, RAM | TBD |
| Source footage and authorization | TBD |
| Camera angle, resolution, FPS | TBD |
| Model names, versions, weights and licenses | TBD |
| Sample size and human annotations | TBD |
| Zone classification agreement | TBD |
| False zone attention and unknown rate | TBD |
| FPS, latency, peak RAM, power/cost | TBD |
| Impact on production pipeline | None: isolated experiment |
| Decision and rationale | TBD |

## Gates

**No-go** if legal rights to footage/weights are unclear, if a gaze heatmap systematically misattributes people to shelves, if unknown predictions are presented as certain, or if processing costs make the proposed retail price uneconomic.

**Go to next stage** only with documented repeatability, manual review of failures, and explicit approval for any new dependency or integration.
