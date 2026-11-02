# Tarbiyat (تربیت)

**A model-training playground where the network is actually written out by hand.**

![Python](https://img.shields.io/badge/python-3.11-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-only%2C%20no%20torch-013243?logo=numpy&logoColor=white)
![Tests](https://img.shields.io/badge/tests-38%20passing-brightgreen)
![License](https://img.shields.io/badge/license-MIT-blue)

---

Upload a CSV, choose a few numbers, hit train, and watch the loss actually fall —
epoch by epoch, from a network whose forward pass, backprop and optimizers are all
plain NumPy in `app/ml/nn.py`.

There's no PyTorch, no TensorFlow, no scikit-learn. That's the entire point. Most
"ML playground" projects put a gradient-descent-shaped UI over a stub that returns
`{"accuracy": 0.94}`. This one computes a real gradient, checks it against finite
differences, and reports what it actually scored — including when that's worse than
a baseline.

## What it does

- **Upload any CSV.** Columns are profiled automatically: numeric columns get
  z-scored, categorical columns get one-hot encoded, and the target column decides
  whether you're doing classification or regression.
- **Pick an architecture.** 1–4 hidden layers, any width per layer (max 256), ReLU/tanh/sigmoid,
  He or Xavier init.
- **Pick an optimizer.** SGD, SGD with momentum, or Adam — with learning rate, batch
  size, L2 penalty, epoch count and seed all exposed.
- **Watch it train.** Per-epoch train loss and accuracy come back in the response; the
  UI draws the curve on a canvas as the run finishes.
- **Read honest metrics.** Accuracy *next to the majority-class baseline*, a confusion
  matrix, per-class precision/recall/F1. For regression: RMSE, MAE, R², all next to a
  mean-predictor baseline. A model that can't beat the baseline is reported as such.
- **Score new rows.** `POST /api/runs/{id}/predict` uses the exact encoding captured
  at training time, so a new row is scaled and one-hot encoded the same way the
  training rows were.
- **Three datasets seeded** so you can try it without finding a CSV first.

## Screenshots

The app ships three views: the run builder, the live loss curve with metrics, and the
confusion matrix. Screenshots aren't committed to this repo — they're generated from
the running app, and a stale screenshot is worse than none. Run it locally and you'll
see the same thing in about a minute.

## Quick start

```bash
git clone https://github.com/mohdabrarbaloch-arch/day-9-tarbiyat.git
cd day-9-tarbiyat
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env          # defaults work as-is for local dev
python3 -m app.seed           # creates the demo user + 3 datasets
uvicorn app.main:app --reload
```

Open <http://127.0.0.1:8000> and log in with:

```
demo@tarbiyat.pk
tarbiyat-demo-2026
```

The demo credentials are seeded in development only. In production the seed script
refuses to create a user with a known password.

### With Docker

```bash
docker compose up --build
```

That brings up Postgres 16 and the app on <http://localhost:8000>. Set `SECRET_KEY`
first — the app won't boot in production with the placeholder value.

## What it actually scores

These numbers come from `scripts/smoke.py` against a running server, not from a
notebook. Reproduce them yourself with `python3 scripts/smoke.py`.

| Dataset | Task | Result | Baseline |
|---|---|---|---|
| Iris (150 rows) | 3-class classification | **97.2%** accuracy | 33.3% (majority) |
| Churn (400 rows) | binary classification | **69.0%** accuracy | 63.0% (majority) |
| Exam score (300 rows) | regression | **R² 0.817**, RMSE 2.19 | RMSE 5.13 (mean predictor) |

Iris is easy and it should be near-perfect. Churn at 69% over a 63% baseline is a
much thinner margin — the synthetic data has real noise in it, and I'd rather show a
small honest gain than a suspiciously round 95%. Exam score lands at R² 0.82 because
the generator adds 2.4 points of Gaussian noise to the target; that's the ceiling,
not a limitation of the network.

## The part worth reading

`app/ml/nn.py` is where the project earns its name. It implements:

- **Forward pass** through an arbitrary stack of dense layers.
- **Analytic backprop**, with the output delta fused per loss so the chain rule is
  correct rather than approximately correct.
- **A correct softmax Jacobian** for MSE over a softmax head — a case most
  from-scratch implementations get wrong by treating the softmax as if it weren't
  coupled across classes.
- **A linear regression head**, because a sigmoid output cannot represent a negative
  z-scored target and silently caps the achievable R².

`scripts/gradcheck.py` compares every analytical gradient against a central finite
difference. All 10 loss/output/head combinations agree to ~1e-10:

```
$ python3 scripts/gradcheck.py
PASS  loss=cce  out=3 act=relu    max_rel_err=1.393e-10
PASS  loss=cce  out=4 act=tanh    max_rel_err=2.197e-10
PASS  loss=cce  out=3 act=sigmoid max_rel_err=2.290e-10
PASS  loss=bce  out=1 act=relu    max_rel_err=8.841e-11
PASS  loss=bce  out=1 act=sigmoid max_rel_err=1.126e-10
PASS  loss=bce  out=1 act=tanh    max_rel_err=1.234e-10
PASS  loss=mse  out=1 act=tanh    max_rel_err=4.801e-11
PASS  loss=mse  out=1 act=relu    max_rel_err=3.566e-11
PASS  loss=mse  out=3 act=relu    max_rel_err=4.032e-11
PASS  loss=mse  out=3 act=tanh    max_rel_err=4.812e-11

10/10 combinations passed (tol=1e-06)
```

That check is what caught two real bugs during development: an output delta computed
against the activation instead of the pre-activation, and the missing softmax
Jacobian. Both produced code that *trained* and looked fine, which is exactly why the
check exists.

## Project layout

```
app/
  ml/nn.py           the network: layers, backprop, optimizers, metrics
  ml/dataset.py      CSV profiling, encoding, splits, EncodingSpec
  services/training.py   orchestration + what the API is allowed to run
  api/routes/        auth, datasets, runs
  models.py          ORM: users, datasets, runs
  schemas.py         request/response validation
  main.py            app factory, middleware, error handling
scripts/
  gradcheck.py       finite-difference verification of the gradients
  smoke.py           end-to-end check against a live server
tests/               38 tests
```

## API

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/health` | liveness |
| `POST` | `/api/auth/register` | create an account |
| `POST` | `/api/auth/login` | get a bearer token |
| `GET` | `/api/auth/me` | current user |
| `GET` | `/api/datasets` | list your datasets |
| `POST` | `/api/datasets` | upload a CSV |
| `GET` | `/api/datasets/{id}` | column profile + preview |
| `DELETE` | `/api/datasets/{id}` | delete |
| `GET` | `/api/runs` | list your runs |
| `POST` | `/api/runs` | train (synchronous, returns the finished run) |
| `GET` | `/api/runs/{id}` | run detail with full history |
| `POST` | `/api/runs/{id}/predict` | score new rows |
| `DELETE` | `/api/runs/{id}` | delete |

Full request/response shapes in [`docs/api.md`](docs/api.md).

## Testing

```bash
pip install -r requirements-dev.txt
pytest -q                 # 38 tests
python3 scripts/gradcheck.py
ruff check . && black --check .
```

## Known limits

Being straight with you about what this is not:

- **Training is synchronous.** A run blocks its request until it finishes. Fine for
  the row and epoch ceilings here (20k rows, 400 epochs); a real training service
  would hand off to a worker queue.
- **The rate limiter is in-process.** It resets on restart and doesn't coordinate
  across workers. Redis if you scale past one.
- **No GPU, no autograd.** Every gradient is hand-derived, which is the point but
  also the ceiling — this is a teaching and demo tool, not a training platform.
- **Binary `.npz`-style weights live in a TEXT column.** Weights round-trip as JSON,
  which is readable and portable but not small. A real system stores a blob.
- **Not deployed.** No Vercel token in the environment that built this. There's no
  live URL because there isn't a live deployment, and I'm not going to invent one.

## License

MIT — see [LICENSE](LICENSE).
