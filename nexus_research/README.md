# Nexus Sight | Attention Intelligence — isolated research

**Status:** v0.0 scaffold; not validated and not production-ready.

- Source fork: https://github.com/RenanCOliveira93/brandpose
- Original project: https://github.com/escudero/brandpose by Claudio Escudero
- Baseline fork commit (before research branch): `9e9665c2601409f3b23da82256e662dede07f5d8`
- Research branch: `nexus/research-attention-v0`
- Production repo `RenanCOliveira93/retail-solutions` remains independent.

## Scope

Build and validate *estimated visual attention per store zone* as a complementary metric to footfall and dwell time. This is NOT verified eye tracking, purchase intent, or exact product recognition.

The first deliverable is an **offline, dependency-free aggregation prototype** operating on synthetic observation records. It does not import BrandPose models, touch CCTV, call APIs, write to databases, or modify upstream scripts.

## Run the first checks

```bash
python -m unittest discover -s nexus_research/tests -v
```

No pip install, network, credentials, videos or model weights are needed.

## Versioned plan

| Stage | Goal | Exit criterion |
|---|---|---|
| v0.0 | Isolation, licenses, baseline, synthetic aggregator | Tests pass; main remains unchanged |
| v0.1 | Reproduce upstream offline on permitted footage | Repeatable setup, fixed versions, performance log |
| v0.2 | Independent gaze adapter | Structured outputs and abstention on uncertain cases |
| v0.3 | Map gaze to configured zones | Manual validation and error analysis |
| v0.4 | Hardware/camera benchmark | Latency, FPS, CPU/GPU, RAM, costs measured |
| v0.5 | Standalone demo and interviews | Retailer feedback, clearly labeled estimates |
| v1.0 candidate | Decide whether to pilot | License, privacy, reliability, economics and business gates passed |

## Non-negotiable boundaries

1. No modifications to `retail-solutions`, its Supabase migrations, production tracker, deployment or portal.
2. No merges to the fork's `main` without explicit review.
3. No automatic dependency on upstream scripts from the production project.
4. Do not store faces or shopper identities. Synthetic data first; any real footage requires legal/privacy review and authorization.
5. Respect upstream MIT attribution and **separate** third-party licenses. YOLO11/Ultralytics AGPL-3.0; Depth Anything V2 Metric Indoor Base CC-BY-NC-4.0; Gaze-LLE MIT per upstream notices. Do not deploy commercially before legal review.
6. The original video is copyright ACTi Corporation; publicly viewable is not equivalent to unrestricted redistribution.
7. Record reproducibility, false positives, abstentions and uncertainty before any commercial claims.

## v0.0 deliverables

- `attention_metrics.py`: deterministic offline zone aggregation.
- `tests/test_attention_metrics.py`: synthetic correctness and edge-case tests.
- `EXPERIMENT_LOG.md`: reproducibility, benchmarks and decision template.

## Next step

Review and execute the tests locally, then reproduce the upstream demo **offline** in a separate virtual environment. No model performance or camera compatibility has been verified by this scaffold.
