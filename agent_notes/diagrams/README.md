# GlucoFoundry — Research Workflow (at a glance)

This is a high-level, general-audience view of how the GlucoFoundry toolkit works
and how it is designed to be **extended**. Every experiment fork — Nocturnal,
Metabonet, the new Meal-Image project, and any future project — reuses this same
pipeline.

![GlucoFoundry research workflow](./glucofoundry_workflow.svg)

> The diagram above is a scalable vector (`glucofoundry_workflow.svg`). A raster
> copy (`glucofoundry_workflow.png`) is also provided for contexts that cannot
> display SVG.

## The workflow in one line

**Datasets → Preprocessing & Features → Training → Evaluation → Reproducible
Results**, with **Models** plugged into training and **Hyperparameter
Optimization (Optuna) + tracking (MLflow)** tuning the models along the way.

```mermaid
flowchart LR
    classDef io fill:#CFF3EE,stroke:#2C7A7B,color:#1D4044,stroke-width:1.5px;
    classDef core fill:#D6E4FF,stroke:#2B6CB0,color:#1A365D,stroke-width:1.5px;
    classDef model fill:#D7F5DD,stroke:#2F855A,color:#1C4532,stroke-width:1.5px;
    classDef ext fill:#FFF4D6,stroke:#D69E2E,color:#744210,stroke-width:1.5px,stroke-dasharray:6 4;

    D[("Datasets<br/>CGM / glucose time series")]:::io
    P["Preprocessing &<br/>Feature Engineering"]:::core
    T["Training /<br/>Fine-tuning"]:::core
    H["Hyperparameter Optimization<br/>Optuna + MLflow tracking"]:::core
    E["Evaluation &<br/>Metrics"]:::core
    R[("Experiments &<br/>Reproducible Results")]:::io
    M["Models<br/>foundation models via<br/>the ModelAdapter protocol"]:::model

    D --> P --> T --> E --> R
    M --> T
    T -- tune --> H
    H -- best config --> E

    xD(["+ Add a dataset"]):::ext -.-> D
    xM(["+ Add a model"]):::ext -.-> M
    xT(["+ Add an experiment / task"]):::ext -.-> T
    xE(["+ Add a metric"]):::ext -.-> E
```

## Built to be extended

The dashed boxes are the toolkit's **extension points** — you add capability
without rewriting the pipeline:

- **Add a dataset** — point the loaders at a new data source.
- **Add a model** — plug any model in through the `ModelAdapter` protocol.
- **Add an experiment / task** — forecasting, classification, event detection,
  and more; the toolkit is task-general, not forecasting-only.
- **Add a metric** — register new evaluation criteria.

## Regenerating the image

The PNG/SVG are produced by a small matplotlib script (only dependency is
matplotlib):

```bash
python agent_notes/diagrams/render_glucofoundry_workflow.py
```

The Mermaid source (`glucofoundry_workflow.mmd`) is the same diagram in a form
that renders natively on GitHub and can be exported with `mmdc` if the Mermaid
CLI is installed.
