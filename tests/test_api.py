"""End-to-end API tests: auth, dataset upload, training, prediction, isolation."""

from __future__ import annotations


def upload_iris(client, headers, csv_text: str, target: str = "label", name: str = "toy"):
    files = {"file": ("toy.csv", csv_text.encode("utf-8"), "text/csv")}
    data = {"target": target, "name": name}
    return client.post("/api/datasets", files=files, data=data, headers=headers)


def test_health_is_public(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_register_then_login_returns_token(client):
    registration = client.post(
        "/api/auth/register",
        json={"email": "new@example.com", "password": "sunda-lambda-7"},
    )
    assert registration.status_code == 201
    token = registration.json()["access_token"]
    assert token.count(".") == 2

    login = client.post(
        "/api/auth/login",
        json={"email": "new@example.com", "password": "sunda-lambda-7"},
    )
    assert login.status_code == 200
    assert login.json()["user"]["email"] == "new@example.com"


def test_duplicate_email_is_rejected(client):
    payload = {"email": "dupe@example.com", "password": "sunda-lambda-7"}
    assert client.post("/api/auth/register", json=payload).status_code == 201
    again = client.post("/api/auth/register", json=payload)
    assert again.status_code == 409


def test_weak_password_is_rejected(client):
    response = client.post(
        "/api/auth/register", json={"email": "weak@example.com", "password": "12345678"}
    )
    assert response.status_code == 422


def test_wrong_password_is_401(client, auth_headers):
    response = client.post(
        "/api/auth/login",
        json={"email": "tester@example.com", "password": "not-the-password"},
    )
    assert response.status_code == 401


def test_me_requires_a_token(client):
    assert client.get("/api/auth/me").status_code == 401


def test_me_returns_current_user(client, auth_headers):
    response = client.get("/api/auth/me", headers=auth_headers)
    assert response.status_code == 200
    assert response.json()["email"] == "tester@example.com"


def test_upload_profiles_columns(client, auth_headers, iris_csv_text):
    response = upload_iris(client, auth_headers, iris_csv_text)
    assert response.status_code == 201, response.text
    body = response.json()
    assert body["row_count"] == 40
    assert body["column_count"] == 3
    assert body["task"] == "classification"
    names = {c["name"] for c in body["profile"]}
    assert {"x", "y", "label"} <= names


def test_upload_requires_target_column(client, auth_headers, iris_csv_text):
    response = upload_iris(client, auth_headers, iris_csv_text, target="missing_column")
    assert response.status_code == 422
    assert "missing_column" in response.json()["detail"]


def test_upload_rejects_empty_file(client, auth_headers):
    files = {"file": ("empty.csv", b"", "text/csv")}
    response = client.post(
        "/api/datasets", files=files, data={"target": "label"}, headers=auth_headers
    )
    assert response.status_code == 400


def test_upload_rejects_header_only_csv(client, auth_headers):
    files = {"file": ("thin.csv", b"a,b,label\n", "text/csv")}
    response = client.post(
        "/api/datasets", files=files, data={"target": "label"}, headers=auth_headers
    )
    assert response.status_code == 422


def test_train_classifier_end_to_end(client, auth_headers, iris_csv_text):
    dataset = upload_iris(client, auth_headers, iris_csv_text).json()

    response = client.post(
        "/api/runs",
        json={
            "dataset_id": dataset["id"],
            "hidden_layers": [8],
            "activation": "tanh",
            "optimizer": "adam",
            "learning_rate": 0.08,
            "epochs": 120,
            "batch_size": 8,
            "test_fraction": 0.25,
            "seed": 7,
        },
        headers=auth_headers,
    )
    assert response.status_code == 201, response.text
    run = response.json()

    assert run["status"] == "completed"
    assert len(run["history"]) >= 1
    assert run["param_count"] > 0
    metrics = run["metrics"]
    assert metrics["task"] == "classification"
    # The toy set is separable, so a working network should clear 80%.
    assert metrics["accuracy"] >= 0.8, metrics
    assert 0.0 <= metrics["majority_baseline"] <= 1.0


def test_regression_run_reports_r2(client, auth_headers):
    rows = ["a,b,target"]
    for i in range(60):
        rows.append(f"{i * 0.1:.2f},{i * 0.05:.2f},{2.5 * i * 0.1 + 0.4:.3f}")
    csv_text = "\n".join(rows) + "\n"

    dataset = upload_iris(
        client, auth_headers, csv_text, target="target", name="regression-toy"
    ).json()
    assert dataset["task"] == "regression"

    response = client.post(
        "/api/runs",
        json={
            "dataset_id": dataset["id"],
            "hidden_layers": [12],
            "activation": "tanh",
            "optimizer": "adam",
            "learning_rate": 0.03,
            "epochs": 150,
            "batch_size": 8,
            "test_fraction": 0.25,
            "seed": 3,
        },
        headers=auth_headers,
    )
    assert response.status_code == 201, response.text
    metrics = response.json()["metrics"]
    assert metrics["task"] == "regression"
    assert "r2" in metrics and "rmse" in metrics
    # A near-linear target with three healthy layers should be fitted well.
    assert metrics["r2"] > 0.7, metrics


def test_regression_run_rejects_a_classification_loss(client, auth_headers):
    """A continuous target cannot be trained with cross-entropy; the API must refuse."""
    rows = ["a,b,target"]
    for i in range(40):
        rows.append(f"{i * 0.1:.2f},{i * 0.2:.2f},{i * 0.37 + 1.1:.3f}")
    csv_text = "\n".join(rows) + "\n"
    dataset = upload_iris(client, auth_headers, csv_text, target="target").json()
    assert dataset["task"] == "regression"

    response = client.post(
        "/api/runs",
        json={"dataset_id": dataset["id"], "hidden_layers": [6], "loss": "cce", "epochs": 5},
        headers=auth_headers,
    )
    assert response.status_code == 422
    assert "cce" in response.json()["detail"].lower()


def test_run_requires_known_dataset(client, auth_headers):
    response = client.post(
        "/api/runs",
        json={"dataset_id": 999_999, "hidden_layers": [4], "epochs": 2},
        headers=auth_headers,
    )
    assert response.status_code == 404


def test_predict_returns_a_label_and_probabilities(client, auth_headers, iris_csv_text):
    dataset = upload_iris(client, auth_headers, iris_csv_text).json()
    run = client.post(
        "/api/runs",
        json={
            "dataset_id": dataset["id"],
            "hidden_layers": [8],
            "activation": "tanh",
            "epochs": 120,
            "batch_size": 8,
            "seed": 5,
        },
        headers=auth_headers,
    ).json()

    response = client.post(
        f"/api/runs/{run['id']}/predict",
        json={"rows": [{"x": 0.2, "y": 0.25}]},
        headers=auth_headers,
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["task"] == "classification"
    prediction = body["predictions"][0]
    assert prediction["predicted"] in {"low", "high"}
    assert abs(sum(prediction["probabilities"].values()) - 1.0) < 1e-6
    # A point far in the low cluster should be classified low.
    assert prediction["predicted"] == "low"


def test_predict_empty_rows_is_rejected(client, auth_headers, iris_csv_text):
    dataset = upload_iris(client, auth_headers, iris_csv_text).json()
    run = client.post(
        "/api/runs",
        json={"dataset_id": dataset["id"], "hidden_layers": [4], "epochs": 30},
        headers=auth_headers,
    ).json()

    response = client.post(
        f"/api/runs/{run['id']}/predict", json={"rows": []}, headers=auth_headers
    )
    assert response.status_code == 400


def test_another_users_dataset_is_invisible(client, auth_headers, iris_csv_text):
    dataset = upload_iris(client, auth_headers, iris_csv_text).json()

    other = client.post(
        "/api/auth/register",
        json={"email": "intruder@example.com", "password": "sunda-lambda-7"},
    ).json()
    other_headers = {"Authorization": f"Bearer {other['access_token']}"}

    response = client.get(f"/api/datasets/{dataset['id']}", headers=other_headers)
    # 404, not 403: a stranger should not learn that this id exists.
    assert response.status_code == 404

    listed = client.get("/api/datasets", headers=other_headers).json()
    assert listed == []


def test_weights_endpoint_exposes_shapes(client, auth_headers, iris_csv_text):
    dataset = upload_iris(client, auth_headers, iris_csv_text).json()
    run = client.post(
        "/api/runs",
        json={"dataset_id": dataset["id"], "hidden_layers": [5], "epochs": 20},
        headers=auth_headers,
    ).json()

    response = client.get(f"/api/runs/{run['id']}/weights", headers=auth_headers)
    assert response.status_code == 200
    body = response.json()
    assert len(body["weights"]) == 2  # one hidden layer + the output layer
    assert body["param_count"] == run["param_count"]


def test_security_headers_are_present(client):
    response = client.get("/api/health")
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["x-frame-options"] == "DENY"
    assert "default-src 'self'" in response.headers["content-security-policy"]


def test_delete_dataset_removes_its_runs(client, auth_headers, iris_csv_text):
    dataset = upload_iris(client, auth_headers, iris_csv_text).json()
    client.post(
        "/api/runs",
        json={"dataset_id": dataset["id"], "hidden_layers": [4], "epochs": 10},
        headers=auth_headers,
    )

    assert client.delete(f"/api/datasets/{dataset['id']}", headers=auth_headers).status_code == 204
    assert client.get("/api/datasets", headers=auth_headers).json() == []
    # Leaving orphaned runs behind would make the runs list lie about its own history.
    assert client.get("/api/runs", headers=auth_headers).json() == []
