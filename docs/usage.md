# Usage

A walk through the app as a person actually uses it.

## 1. Get in

Open the app, register an account (or use the seeded `demo@tarbiyat.pk` /
`tarbiyat-demo-2026`). The token is held in `localStorage` and the SPA redirects you to
the workspace.

## 2. Pick or upload a dataset

Three datasets are already there so you don't need to hunt for a CSV on your first run:

| Dataset | Rows | Task | What it's for |
|---|---|---|---|
| Iris species (3-class) | 150 | classification | the "does this even work" sanity check |
| Customer churn (binary) | 400 | classification | mixed numeric + categorical columns |
| Exam score (regression) | 300 | regression | a continuous target, different metrics |

To bring your own: **Upload CSV**, choose the file, and pick the target column. The
app profiles every column and tells you what it found — which columns it treated as
numeric, which as categorical, and how many values were missing.

A CSV needs a header row and at least two columns. The target column can be text
(classification) or numeric (regression) — the app decides from the data, and you'll see
which it chose before you train.

## 3. Build a run

The builder exposes four groups:

**Architecture** — hidden layers as a comma list, e.g. `12,8` for two layers. Every
layer is dense with the chosen activation. Output width is set for you from the task:
1 for regression or binary, *n* for *n* classes.

**Optimizer** — SGD, momentum, or Adam. Adam at `0.01` is a fine default for these
sizes; SGD usually wants something closer to `0.1`.

**Training** — epochs, batch size, test fraction, L2, seed. The seed makes runs
reproducible: same data + same config + same seed gives the same weights.

**Loss** — leave it on *auto* unless you have a reason. Auto picks cross-entropy for
classification and MSE for regression.

## 4. Read the result honestly

This is the part worth slowing down for.

**Classification** shows accuracy *next to the majority-class baseline*. If 63% of your
customers churned and your model scores 69%, that's a real but modest gain — and the
app says so rather than celebrating 69%. The confusion matrix shows where it's wrong,
and per-class precision/recall/F1 shows whether one class is carrying the average.

**Regression** shows RMSE and R² *next to a mean-predictor baseline*. An R² of 0.82
means you explain 82% of the variance; the remaining 18% is noise the generator put
there on purpose, and no amount of extra epochs will recover it.

**The loss curve** is the fastest way to spot trouble:

- Falls smoothly, plateaus → healthy.
- Falls then rises on validation while train keeps falling → overfitting. Fewer
  layers, more L2, fewer epochs.
- Flat from epoch 1 → learning rate too small, or the features carry no signal.
- Wild oscillation → learning rate too high.

## 5. Score new rows

On a finished run, **Predict** takes a JSON row and returns the label with a confidence
and the full probability distribution. The row is encoded with the same spec used in
training — same z-score means, same one-hot columns — so what you get back matches what
the model actually learned.

An unseen category (say `plan: "enterprise"` when training only saw basic/premium/
standard) encodes as "none of the above" rather than as some arbitrary existing class.

## Worked example: finding the churn threshold

Train the churn dataset with `12` hidden units, Adam at `0.01`, 120 epochs. You'll land
around 69% accuracy against a 63% baseline.

Now predict two customers:

```json
{ "tenure_months": 2,  "monthly_charges": 3200, "support_tickets": 8,
  "plan": "basic", "contract": "monthly" }
```

```json
{ "tenure_months": 48, "monthly_charges": 5100, "support_tickets": 0,
  "plan": "premium", "contract": "annual" }
```

The first comes back `1` (churned) at high confidence; the second `0`. Short tenure,
monthly contract and support tickets are the signals — which is also what you'd guess
from domain knowledge, and that agreement is the useful part.

## Reproducing runs from the command line

```bash
TOKEN=$(curl -s -X POST localhost:8000/api/auth/login \
  -H 'Content-Type: application/json' \
  -d '{"email":"demo@tarbiyat.pk","password":"tarbiyat-demo-2026"}' \
  | python3 -c 'import sys,json; print(json.load(sys.stdin)["access_token"])')

curl -s -X POST localhost:8000/api/runs \
  -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' \
  -d '{"dataset_id":1,"hidden_layers":[12,8],"epochs":120,"seed":42}' \
  | python3 -m json.tool
```

Or just run `python3 scripts/smoke.py`, which does the whole flow and asserts the
results beat their baselines.
