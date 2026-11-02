"""Live smoke test against a running server. Verifies the flow a real user takes.

Run:  python3 scripts/smoke.py [base_url]
"""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8009"
FAILURES: list[str] = []


def call(method: str, path: str, body: dict | None = None, token: str | None = None):
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(BASE + path, data=data, method=method)
    request.add_header("Content-Type", "application/json")
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            return response.status, json.loads(response.read() or b"null")
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read() or b"null")


def check(label: str, condition: bool, detail: str = "") -> None:
    mark = "PASS" if condition else "FAIL"
    print(f"[{mark}] {label}{(' - ' + detail) if detail else ''}")
    if not condition:
        FAILURES.append(label)


def main() -> int:
    status, health = call("GET", "/api/health")
    check("health returns 200 ok", status == 200 and health.get("status") == "ok", str(health))

    status, login = call(
        "POST",
        "/api/auth/login",
        {"email": "demo@tarbiyat.pk", "password": "tarbiyat-demo-2026"},
    )
    check("demo login succeeds", status == 200, f"status {status}")
    token = login["access_token"]

    status, me = call("GET", "/api/auth/me", token=token)
    check("me returns the demo user", status == 200 and me["email"] == "demo@tarbiyat.pk")

    status, datasets = call("GET", "/api/datasets", token=token)
    check("three seeded datasets", status == 200 and len(datasets) == 3, f"{len(datasets)} found")
    for dataset in datasets:
        print(
            f"        dataset {dataset['id']}: {dataset['name']} "
            f"[{dataset['task']}] {dataset['row_count']} rows x {dataset['column_count']} cols"
        )

    by_name = {d["name"]: d for d in datasets}
    iris = by_name["Iris species (3-class)"]
    churn = by_name["Customer churn (binary)"]
    score = by_name["Exam score (regression)"]

    # --- classification ---
    status, run = call(
        "POST",
        "/api/runs",
        {
            "dataset_id": iris["id"],
            "hidden_layers": [12, 8],
            "activation": "relu",
            "optimizer": "adam",
            "learning_rate": 0.01,
            "epochs": 120,
            "batch_size": 16,
            "test_fraction": 0.25,
            "seed": 42,
        },
        token,
    )
    if status != 201:
        check("classification training", False, f"status {status}: {run}")
    else:
        metrics = run["metrics"]
        print(
            f"        run {run['id']}: {run['param_count']} params, "
            f"{len(run['history'])} epochs, {run['duration_ms']} ms"
        )
        print(
            f"        accuracy {metrics['accuracy']} vs majority baseline "
            f"{metrics['majority_baseline']}, labels {metrics['labels']}"
        )
        print(f"        confusion {metrics['confusion_matrix']}")
        check(
            "classification beats the majority baseline",
            metrics["accuracy"] > metrics["majority_baseline"],
            f"{metrics['accuracy']} vs {metrics['majority_baseline']}",
        )
        check(
            "loss history recorded per epoch",
            len(run["history"]) == 120
            and run["history"][-1]["train_loss"] < run["history"][0]["train_loss"],
        )

        status, prediction = call(
            "POST",
            f"/api/runs/{run['id']}/predict",
            {
                "rows": [
                    {
                        "sepal_length": 5.0,
                        "sepal_width": 3.4,
                        "petal_length": 1.5,
                        "petal_width": 0.2,
                    }
                ]
            },
            token,
        )
        if status != 200:
            check("predict an iris row", False, f"status {status}: {prediction}")
        else:
            first = prediction["predictions"][0]
            print(f"        prediction: {first['predicted']} @ {first['confidence']}")
            check(
                "setosa row predicted as setosa",
                first["predicted"] == "setosa",
                f"got {first['predicted']}",
            )
            check(
                "probabilities sum to 1",
                abs(sum(first["probabilities"].values()) - 1.0) < 1e-3,
            )

    # --- a binary target with categorical features ---
    status, churn_run = call(
        "POST",
        "/api/runs",
        {
            "dataset_id": churn["id"],
            "hidden_layers": [12],
            "activation": "relu",
            "optimizer": "adam",
            "learning_rate": 0.01,
            "epochs": 120,
            "batch_size": 16,
            "test_fraction": 0.25,
            "seed": 42,
        },
        token,
    )
    if status != 201:
        check("churn training", False, f"status {status}: {churn_run}")
    else:
        m = churn_run["metrics"]
        print(
            f"        churn accuracy {m['accuracy']} vs baseline {m['majority_baseline']}, "
            f"labels {m['labels']}"
        )
        check(
            "churn beats its majority baseline",
            m["accuracy"] > m["majority_baseline"],
            f"{m['accuracy']} vs {m['majority_baseline']}",
        )
        # A short-tenure, ticket-heavy, monthly-contract customer is the textbook churner.
        status, churn_pred = call(
            "POST",
            f"/api/runs/{churn_run['id']}/predict",
            {
                "rows": [
                    {
                        "tenure_months": 2,
                        "monthly_charges": 3200,
                        "support_tickets": 8,
                        "plan": "basic",
                        "contract": "monthly",
                    }
                ]
            },
            token,
        )
        if status != 200:
            check("predict a churn row", False, f"status {status}: {churn_pred}")
        else:
            first = churn_pred["predictions"][0]
            print(f"        churn prediction: {first['predicted']} @ {first['confidence']}")
            check(
                "high-risk row is labelled churned", first["predicted"] == "1", first["predicted"]
            )

    # --- regression ---
    status, run2 = call(
        "POST",
        "/api/runs",
        {
            "dataset_id": score["id"],
            "hidden_layers": [16, 8],
            "activation": "tanh",
            "optimizer": "adam",
            "learning_rate": 0.01,
            "epochs": 200,
            "batch_size": 16,
            "test_fraction": 0.25,
            "seed": 7,
        },
        token,
    )
    if status != 201:
        check("regression training", False, f"status {status}: {run2}")
    else:
        m = run2["metrics"]
        print(f"        r2 {m['r2']}, rmse {m['rmse']}, baseline rmse {m['baseline_rmse']}")
        check("regression r2 beats 0.6", m["r2"] > 0.6, f"r2={m['r2']}")
        check("regression rmse beats baseline", m["rmse"] < m["baseline_rmse"])

    # --- validation the API must refuse ---
    status, bad = call(
        "POST",
        "/api/runs",
        {"dataset_id": score["id"], "hidden_layers": [6], "loss": "cce", "epochs": 3},
        token,
    )
    check("regression + cce is refused", status == 422, f"status {status}")

    status, bad2 = call("POST", "/api/runs", {"dataset_id": 999999, "hidden_layers": [4]}, token)
    check("unknown dataset is 404", status == 404, f"status {status}")

    # --- the SPA must actually be served ---
    request = urllib.request.Request(BASE + "/")
    with urllib.request.urlopen(request, timeout=30) as response:
        html = response.read().decode()
    check("index.html served at /", response.status == 200 and "Tarbiyat" in html)
    check("app.js linked", "/static/app.js" in html)

    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILED: {', '.join(FAILURES)}")
        return 1
    print("All smoke checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
