"""Seed the demo account and three starter datasets.

The data is generated deterministically from a fixed seed, so it is synthetic but
internally consistent — every target really is a function of its features plus noise.
That matters: a demo dataset with no learnable signal would make the whole playground
look broken.

Run with:  python -m app.seed
"""

from __future__ import annotations

import csv
import io
import json
import math

from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.security import hash_password
from app.db.base import SessionLocal, init_db
from app.ml.dataset import parse_csv
from app.models import Dataset, User

# A four-class iris-like dataset with overlapping clusters. Overlap is intentional —
# a perfectly separable demo would teach the wrong lesson about what accuracy means.
IRIS_CENTERS = [
    (5.00, 3.40, 1.46, 0.24, "setosa"),
    (5.90, 2.75, 4.25, 1.30, "versicolor"),
    (6.60, 3.00, 5.55, 2.00, "virginica"),
]


def _gauss(rng, mean: float, sd: float) -> float:
    # Box-Muller, so the seed file has no numpy dependency at import time.
    u1 = max(rng.random(), 1e-12)
    u2 = rng.random()
    return mean + sd * math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)


def iris_csv(n_per_class: int = 50, seed: int = 7) -> str:
    import random

    rng = random.Random(seed)
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["sepal_length", "sepal_width", "petal_length", "petal_width", "species"])
    for center in IRIS_CENTERS:
        for _ in range(n_per_class):
            writer.writerow(
                [
                    round(_gauss(rng, center[0], 0.36), 2),
                    round(_gauss(rng, center[1], 0.32), 2),
                    round(_gauss(rng, center[2], 0.42), 2),
                    round(_gauss(rng, center[3], 0.24), 2),
                    center[4],
                ]
            )
    return buffer.getvalue()


def churn_csv(n: int = 400, seed: int = 11) -> str:
    """Binary classification with a genuinely non-trivial relationship."""
    import random

    rng = random.Random(seed)
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(
        ["tenure_months", "monthly_charges", "support_tickets", "plan", "contract", "churned"]
    )
    for _ in range(n):
        tenure = max(1, min(72, int(_gauss(rng, 28, 16))))
        charges = max(300, round(_gauss(rng, 2400, 900), 0))
        tickets = max(0, min(9, int(_gauss(rng, 2.2, 1.7))))
        plan = rng.choice(["basic", "standard", "premium"])
        contract = rng.choice(["monthly", "annual"])

        # The real relationship: short tenure, many tickets and a monthly contract all
        # push churn up. A model that ignores these should land near the base rate.
        logit = (
            -1.15
            - 0.045 * tenure
            + 0.62 * tickets
            + 0.00019 * charges
            + (0.85 if contract == "monthly" else -0.35)
            + (0.30 if plan == "basic" else -0.15)
        )
        p = 1.0 / (1.0 + math.exp(-logit))
        writer.writerow([tenure, charges, tickets, plan, contract, 1 if rng.random() < p else 0])
    return buffer.getvalue()


def score_csv(n: int = 300, seed: int = 13) -> str:
    """Regression target, so the playground can be seen doing more than labelling."""
    import random

    rng = random.Random(seed)
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["hours_studied", "attendance_pct", "past_score", "sleep_hours", "final_score"])
    for _ in range(n):
        hours = max(0.0, round(_gauss(rng, 4.5, 2.1), 1))
        attendance = max(40.0, min(100.0, round(_gauss(rng, 82, 11), 1)))
        past = max(20.0, min(100.0, round(_gauss(rng, 62, 15), 1)))
        sleep = max(4.0, min(10.0, round(_gauss(rng, 7.0, 1.1), 1)))

        score = (
            18.0
            + 4.6 * hours
            + 0.31 * attendance
            + 0.42 * past
            + 1.7 * sleep
            + _gauss(rng, 0.0, 2.4)
        )
        writer.writerow([hours, attendance, past, sleep, round(max(0.0, min(100.0, score)), 1)])
    return buffer.getvalue()


STARTERS: list[tuple[str, str, str, str]] = [
    (
        "Iris species (3-class)",
        "species",
        "Uses four petal and sepal measurements to label the flower species. The classes "
        "overlap on purpose, so you can watch accuracy settle below 100% for honest reasons.",
        "iris",
    ),
    (
        "Customer churn (binary)",
        "churned",
        "A telecom-style churn table. Short tenure, more support tickets and a monthly "
        "contract all raise the odds of leaving. Compare your accuracy against the "
        "majority-class baseline the app reports.",
        "churn",
    ),
    (
        "Exam score (regression)",
        "final_score",
        "Predicts a final score from study hours, attendance, prior marks and sleep. "
        "Continuous target, so the app switches to RMSE and R² automatically. The noise "
        "on the target is real, so a good model lands around R² 0.8 rather than 1.0.",
        "score",
    ),
]


def _upsert(
    session: Session, user: User, name: str, target: str, description: str, body: str
) -> Dataset | None:
    existing = (
        session.query(Dataset).filter(Dataset.owner_id == user.id, Dataset.name == name).first()
    )
    if existing is not None:
        return None

    parsed = parse_csv(body.encode("utf-8"), target=target, max_rows=20_000, max_columns=60)
    dataset = Dataset(
        owner_id=user.id,
        name=name,
        target=target,
        task=parsed.task,
        fingerprint=parsed.fingerprint,
        row_count=len(parsed.rows),
        column_count=len(parsed.columns),
        csv_body=body,
        profile_json=json.dumps([p.as_dict() for p in parsed.profiles]),
    )
    session.add(dataset)
    return dataset


def seed() -> dict[str, object]:
    settings = get_settings()
    init_db()
    session = SessionLocal()
    created: list[str] = []
    try:
        user = session.query(User).filter(User.email == settings.demo_email).first()
        if user is None:
            user = User(
                email=settings.demo_email,
                password_hash=hash_password(settings.demo_password),
                display_name="Demo user",
            )
            session.add(user)
            session.commit()
            session.refresh(user)
            created.append("user")

        bodies = {
            "iris": iris_csv(),
            "churn": churn_csv(),
            "score": score_csv(),
        }
        for name, target, description, key in STARTERS:
            dataset = _upsert(session, user, name, target, description, bodies[key])
            if dataset is not None:
                created.append(name)

        session.commit()
        return {
            "demo_email": settings.demo_email,
            "created": created,
            "datasets": [s[0] for s in STARTERS],
        }
    finally:
        session.close()


if __name__ == "__main__":
    result = seed()
    print(json.dumps(result, indent=2))
