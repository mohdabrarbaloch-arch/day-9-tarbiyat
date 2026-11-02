# Setup

## Requirements

- Python 3.11+
- No GPU, no CUDA, no system math libraries. NumPy ships as a wheel.
- Docker only if you want Postgres without installing it.

## Local, five commands

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
python3 -m app.seed
uvicorn app.main:app --reload
```

Then open <http://127.0.0.1:8000>.

The `.env.example` defaults are deliberately usable as-is: SQLite at `./tarbiyat.db`,
a placeholder secret, development mode. You do not need to edit anything to get a
running app.

## The demo account

`python3 -m app.seed` creates:

| Field | Value |
|---|---|
| Email | `demo@tarbiyat.pk` |
| Password | `tarbiyat-demo-2026` |

plus three datasets: Iris (150 rows), Customer churn (400 rows), Exam score (300 rows).

The seed is idempotent — running it twice won't duplicate anything. It refuses to
create the demo user when `ENVIRONMENT=production`, so a deployed instance never has a
known credential.

## Docker, with Postgres

```bash
cp .env.example .env
# set a real secret before this
SECRET_KEY=$(python3 -c "import secrets; print(secrets.token_urlsafe(48))") \
  docker compose up --build
```

Services:

| Service | Port | Notes |
|---|---|---|
| `app` | 8000 | waits for the database healthcheck |
| `db` | 5432 | Postgres 16, data in the `pgdata` volume |

The container runs as UID 10001, not root, and has a `HEALTHCHECK` hitting
`/api/health`.

## Environment variables

| Variable | Default | What it does |
|---|---|---|
| `APP_NAME` | `Tarbiyat` | Shown in the OpenAPI title |
| `ENVIRONMENT` | `development` | `production` enables the secret-key guard |
| `DEBUG` | `false` | Verbose logging |
| `SECRET_KEY` | dev placeholder | JWT signing key. **Must change in production.** |
| `ALGORITHM` | `HS256` | JWT signing algorithm |
| `ACCESS_TOKEN_TTL_MINUTES` | `1440` | Token lifetime (24 h) |
| `DATABASE_URL` | `sqlite:///./tarbiyat.db` | Any SQLAlchemy URL |
| `CORS_ORIGINS` | `*` | `*` or a comma-separated origin list |
| `MAX_UPLOAD_BYTES` | `5000000` | Upload ceiling (5 MB) |
| `MAX_DATASET_ROWS` | `20000` | Rows considered when profiling/training |
| `MAX_DATASET_COLUMNS` | `60` | Column ceiling for an uploaded CSV |
| `MAX_EPOCHS` | `400` | Hard epoch ceiling, enforced server-side |
| `MAX_HIDDEN_LAYERS` | `4` | Hard layer-count ceiling |
| `MAX_LAYER_WIDTH` | `256` | Widest single hidden layer |
| `MAX_CONCURRENT_RUNS` | `2` | *Declared, not yet enforced* — reserved for a queue build |
| `RUN_TIMEOUT_SECONDS` | `180` | *Declared, not yet enforced* — same |
| `RATE_LIMIT_AUTH` | `10/minute` | Login/register, per IP per path |
| `RATE_LIMIT_WRITE` | `60/minute` | Dataset uploads and training |
| `SEED_DEMO_USER` | `true` | Create the demo account on seed |
| `DEMO_EMAIL` | `demo@tarbiyat.pk` | Demo account email |
| `DEMO_PASSWORD` | `tarbiyat-demo-2026` | Demo password; refused in production |

The two variables marked *not yet enforced* are read into `Settings` but nothing
consults them. They're listed anyway rather than quietly deleted, because a config file
that promises a limit it doesn't apply is worse than one that admits the gap.

Generate a production secret with:

```bash
python3 -c "import secrets; print(secrets.token_urlsafe(48))"
```

## Verifying your install

Two levels, both quick:

```bash
# unit + API tests, no server needed
pip install -r requirements-dev.txt
pytest -q

# gradients vs finite differences
python3 scripts/gradcheck.py

# full flow against a live server
uvicorn app.main:app --port 8009 &
python3 scripts/smoke.py
```

`smoke.py` trains all three seeded datasets and checks the results actually beat their
baselines — so it fails loudly if a refactor quietly breaks learning, which the unit
tests alone would not catch.

## Common problems

**`table runs has no column named encoding_json`**
The database file predates a schema change. There are no migrations in this project
yet, so delete it and re-run: `rm tarbiyat.db && python3 -m app.seed`.

**`RuntimeError: SECRET_KEY must be changed in production`**
You set `ENVIRONMENT=production` without a real secret. That guard is intentional.

**Training returns 422 about the loss**
The requested loss doesn't fit the detected task — e.g. `cce` on a continuous target.
Omit `loss` and the app picks the right one for you.

**Training is slow**
It's single-threaded NumPy. Lower `epochs` or `MAX_TRAINING_ROWS`, or widen the batch.
There's no GPU path by design.
