# Crypto Price Prediction Platform — real-time ML on Kubernetes

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![Rust](https://img.shields.io/badge/Rust-stable-000000?logo=rust)
![Kubernetes](https://img.shields.io/badge/Kubernetes-kind%20%7C%20DigitalOcean-326CE5?logo=kubernetes&logoColor=white)
![Kafka](https://img.shields.io/badge/Kafka-Strimzi-231F20?logo=apachekafka)
![MLflow](https://img.shields.io/badge/MLflow-registry-0194E2?logo=mlflow&logoColor=white)

An end-to-end, production-grade system that ingests live cryptocurrency trades, computes streaming features, trains and registers short-horizon price models, serves predictions through a Rust API, and monitors both data drift and model quality — all deployed on Kubernetes with GitOps.

The point of this repository is not the model. It is everything around the model that makes a prediction trustworthy and operable: streaming, feature stores, experiment tracking, validation, observability, and the cost of running it.

---

## What it does

```
Kraken WebSocket ──► trades ──► Kafka ──► candles ──► Kafka ──► technical_indicators ──► Kafka ──► RisingWave (feature store)
                                                                                                         │
                                   news API ──► news ──► Kafka ──► news-sentiment (LLM) ──► Kafka ───────┤
                                                                                                         ▼
                                                                    training-pipeline (CronJob) ──► MLflow registry
                                                                                                         │
                                                      predictor ──► predictions table ◄── prediction-api (Rust/axum) ──► clients
                                                                                                         │
                                                                 ml-stability exporter ──► Prometheus ──► Grafana
```

| Stage               | Service                             | Role                                                                                                                                  |
| ------------------- | ----------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| Ingestion           | `trades`                            | Streams live trades from the Kraken WebSocket API into Kafka; historical backfill runs as a Kubernetes Job                            |
| Feature engineering | `candles`, `technical_indicators`   | Stateful stream processing (Quix Streams): OHLCV candles, then a configurable set of technical indicators (SMA, EMA, RSI, MACD, OBV…) |
| Context             | `news`, `news-sentiment`            | Pulls market news and scores sentiment with an LLM; published as a Kafka topic                                                        |
| Feature store       | RisingWave                          | Streaming SQL database materialising features as queryable tables                                                                     |
| Training            | `predictor` (training pipeline)     | Scheduled CronJob: data validation, candidate screening, hyperparameter search, model registration                                    |
| Serving             | `prediction-api` (Rust, axum, sqlx) | Low-latency REST endpoint returning the latest prediction per pair; distroless `scratch` image, unprivileged user                     |
| Monitoring          | `ml-stability`                      | Custom Prometheus exporter exposing MLflow test metrics (MAE) as gauges for Grafana alerting                                          |

---

## Engineering highlights

**Streaming and data**

- Exactly-once-style stateful windows for candles, with late-trade handling and backfill via a dedicated Job rather than a long-lived Deployment.
- Technical indicators driven by configuration (`param_space.yaml`, service config), not hard-coded constants.
- Feature store on RisingWave with explicit SQL migrations per service.

**ML lifecycle**

- Candidate screening with LazyPredict, then Optuna hyperparameter optimisation over the top _N_ candidates, selecting the best by out-of-sample MAE.
- MLflow for experiment tracking, model registry and artifact storage (model signature inferred and logged).
- Data-drift report (Evidently `DataDriftPreset`) generated at every training run against the data the current production model was trained on, and logged as an MLflow artifact.
- Model quality exported to Prometheus (`mlflow_model_mae_test`) by a custom exporter, so model degradation is alertable like any other SLO.

**Serving and performance**

- The prediction API is written in Rust (axum + sqlx). A functionally identical FastAPI implementation (`services/prediction_api_py`) exists for comparison.
- Benchmarked under load with Grafana k6 (200 virtual users, 25-minute profile) and profiled with `perf`; CPU flamegraphs for both implementations are in the repository root, and the full method is documented in [`k6-tests/prediction_api`](k6-tests/prediction_api/readme.md).
- Single-request `perf stat`: **7 ms task-clock, 1 context switch (Rust)** vs **16.5 ms, 19 context switches (Python)**. Memory footprint comparable. Caveat documented: micro-benchmark, not a concurrency study.

**Containers and deployment**

- Multi-stage builds throughout. The Rust API ships as a `FROM scratch` image with CA certificates, timezone data and a non-root user: **19 MB instead of 135 MB**. The training pipeline image is pruned (no `.pyc`, no test suites, slim runtime) with `uv` for reproducible, locked installs.
- Images published to GitHub Container Registry; build-and-push and deploy scripts under `scripts/`.
- Two environments: a local `kind` cluster with port-mapping (Kafka/Strimzi, Kafka UI, RisingWave, MLflow, Prometheus, Grafana installed by script), and a **production cluster on DigitalOcean** with Strimzi Kafka, ingress-nginx and **FluxCD** reconciling manifests from this repository. Setup notes in [`deployments/prod`](deployments/prod/README.MD).
- Kustomize overlays per service; the training pipeline runs as both a one-off Job and a CronJob.

**Code quality**

- `uv` workspaces, one `pyproject.toml` per service, shared tooling at the root.
- `ruff` linting via pre-commit.
- Pydantic-settings configuration for every service; secrets injected through Kubernetes rather than baked into images.

---

## Repository layout

```
services/
  trades/                 Kraken WebSocket → Kafka producer (+ historical backfill)
  candles/                Trades → OHLCV candles (Quix Streams)
  technical_indicators/   Candles → indicators → RisingWave
  news/ , news-sentiment/ News ingestion and LLM sentiment scoring
  predictor/              Training pipeline, prediction generator, data validation, Prometheus exporter
  prediction-api/         Rust (axum + sqlx) serving API
  prediction_api_py/      FastAPI equivalent, kept for benchmarking
docker/                   One multi-stage Dockerfile per service (incl. scratch build for the Rust API)
deployments/
  dev/                    kind cluster scripts + Kustomize manifests per service
  prod/                   DigitalOcean cluster, Strimzi Kafka, FluxCD GitOps
dashboards/               Grafana dashboard (candles.json)
k6-tests/                 Load-test scripts, flamegraph and perf methodology
scripts/                  build-and-push-image.sh, deploy.sh
```

---

## Quick start (local)

Prerequisites: Docker, `kind`, `kubectl`, `uv`, Rust toolchain (for the API), a Kraken API key for live trades.

```bash
# 1. Create the local cluster and install the platform (Kafka, RisingWave, MLflow, Prometheus, Grafana)
cd deployments/dev/kind && ./create_cluster.sh && ./install_kafka.sh && ./install_risingwave.sh \
  && ./install_prometheus.sh && ./install_grafana.sh

# 2. Run a service locally against the cluster
make dev service=trades

# 3. Build, push and deploy a service
make build-and-push image_name=trades env=dev
make deploy service=trades env=dev

# 4. Query the prediction API
curl "http://localhost:3001/prediction?pair=ETH/EUR"
# {"pair":"ETH/EUR","predicted_price":3715.34,"ts_ms":1755551431063,"predicted_ts_ms":1755551760000}
```

Each service has its own README with configuration variables.

---

## Results and limitations

- The system runs end to end on both environments; the prediction endpoint returns sub-10 ms CPU time per request in Rust.
- Predictive accuracy is **not** the claim of this project. Short-horizon crypto prices are close to a random walk; the training pipeline reports MAE against a naive baseline and the Prometheus exporter exists precisely to show when the model stops adding value.
- Known gaps, tracked in the roadmap: a single parameterised Dockerfile instead of one per service, OCI image labels and versioning from `pyproject.toml`, Helm packaging, model-drift (not only data-drift) tests, Grafana alert rules, and stricter error handling in the Rust API (`Result` propagation instead of `unwrap`).

---

## Roadmap

- [ ] Single Dockerfile with build arguments for all Python services; image version read from `pyproject.toml`; OCI labels
- [ ] Helm chart for one service, then all
- [ ] Model-drift tests (Evidently `RegressionPreset` with `include_tests=True`) in the training pipeline
- [ ] Grafana provisioning via ConfigMap (dashboard + pair selector) and alert rules on `mlflow_model_mae_test`
- [ ] `prediction_horizon_seconds` and SMA windows fully externalised to config
- [ ] Rust API: `Result<_, sqlx::Error>` propagation, current close price alongside the prediction
- [ ] Evaluate Pandera for schema validation at ingestion

---

## Author

**Karim Abousselham** — civil engineer (EHTP), public-finance magistrate, PhD candidate in AI for power-grid decarbonisation (USMBA).
[LinkedIn](https://www.linkedin.com/in/karim-abousselham) · karim.abousselham@gmail.com

This system started from the architecture taught in Pau Labarta Bajo's _Real-World ML_ program and was extended (Rust serving layer and benchmarking, Prometheus model-quality exporter, production cluster with GitOps, drift reporting, hyperparameter search, image hardening). Licensed under MIT.
