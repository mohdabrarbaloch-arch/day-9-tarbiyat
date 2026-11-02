"""Training runs: launch, list, inspect, delete, predict."""

from __future__ import annotations

import json

from fastapi import APIRouter, HTTPException, Request, status

from app.api.deps import AppSettings, CurrentUser, DbSession
from app.ml.dataset import DatasetError
from app.models import Dataset, TrainingRun
from app.schemas import (
    EpochOut,
    PredictionOut,
    PredictRequest,
    PredictResponse,
    RunOut,
    RunSummary,
    TrainRequest,
)
from app.services.training import TrainingError, predict_rows, run_training

router = APIRouter(prefix="/api/runs", tags=["runs"])


def _owned_run(session: DbSession, user_id: int, run_id: int) -> TrainingRun:
    run = session.get(TrainingRun, run_id)
    if run is None or run.owner_id != user_id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Run not found.")
    return run


def _to_out(run: TrainingRun) -> RunOut:
    return RunOut(
        id=run.id,
        dataset_id=run.dataset_id,
        label=run.label,
        status=run.status,
        error=run.error,
        loss=run.loss,
        config=run.config,
        param_count=run.param_count,
        train_rows=run.train_rows,
        test_rows=run.test_rows,
        duration_ms=run.duration_ms,
        created_at=run.created_at.isoformat(),
        history=[
            EpochOut(
                epoch=row["epoch"],
                train_loss=row["train_loss"],
                train_accuracy=row.get("train_accuracy"),
                val_loss=row.get("val_loss"),
                val_accuracy=row.get("val_accuracy"),
            )
            for row in run.history
        ],
        metrics=run.metrics,
    )


def _summary(run: TrainingRun, dataset_name: str) -> RunSummary:
    metrics = run.metrics or {}
    if metrics.get("task") == "classification":
        score = metrics.get("accuracy")
        kind = "accuracy"
    elif metrics.get("task") == "regression":
        score = metrics.get("r2")
        kind = "r2"
    else:
        score = None
        kind = ""
    return RunSummary(
        id=run.id,
        dataset_id=run.dataset_id,
        dataset_name=dataset_name,
        label=run.label,
        status=run.status,
        loss=run.loss,
        score=score,
        score_kind=kind,
        epochs_run=len(run.history),
        param_count=run.param_count,
        duration_ms=run.duration_ms,
        created_at=run.created_at.isoformat(),
    )


@router.post("", response_model=RunOut, status_code=status.HTTP_201_CREATED)
def start_run(
    payload: TrainRequest,
    session: DbSession,
    user: CurrentUser,
    settings: AppSettings,
    request: Request,
) -> RunOut:
    limiter = getattr(request.app.state, "limiter", None)
    if limiter is not None:
        limiter.check(request, settings.rate_limit_write)

    dataset = session.get(Dataset, payload.dataset_id)
    if dataset is None or dataset.owner_id != user.id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Dataset not found.")

    config = payload.model_dump()
    try:
        outcome = run_training(session, user, dataset, config, settings)
        return _to_out(outcome.run)
    except (TrainingError, DatasetError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from exc
    except MemoryError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="That configuration needs more memory than this deployment allows. "
            "Reduce the layer widths or the epoch count.",
        ) from exc


@router.get("", response_model=list[RunSummary])
def list_runs(
    session: DbSession, user: CurrentUser, limit: int = 50, dataset_id: int | None = None
) -> list[RunSummary]:
    query = session.query(TrainingRun).filter(TrainingRun.owner_id == user.id)
    if dataset_id is not None:
        query = query.filter(TrainingRun.dataset_id == dataset_id)
    runs = query.order_by(TrainingRun.created_at.desc()).limit(min(limit, 200)).all()

    names: dict[int, str] = {}
    dataset_ids = {r.dataset_id for r in runs}
    if dataset_ids:
        for d in session.query(Dataset).filter(Dataset.id.in_(dataset_ids)).all():
            names[d.id] = d.name

    return [_summary(run, names.get(run.dataset_id, "")) for run in runs]


@router.get("/{run_id}", response_model=RunOut)
def get_run(run_id: int, session: DbSession, user: CurrentUser) -> RunOut:
    return _to_out(_owned_run(session, user.id, run_id))


@router.delete("/{run_id}", status_code=status.HTTP_204_NO_CONTENT, response_model=None)
def delete_run(run_id: int, session: DbSession, user: CurrentUser):
    run = _owned_run(session, user.id, run_id)
    session.delete(run)
    session.commit()


@router.post("/{run_id}/predict", response_model=PredictResponse)
def predict(
    run_id: int,
    payload: PredictRequest,
    session: DbSession,
    user: CurrentUser,
    request: Request,
    settings: AppSettings | None = None,
) -> PredictResponse:
    run = _owned_run(session, user.id, run_id)
    dataset = session.get(Dataset, run.dataset_id)
    if dataset is None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="The dataset behind this run was deleted, so it cannot be scored.",
        )
    if not payload.rows:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Provide at least one row."
        )
    if len(payload.rows) > 500:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="At most 500 rows can be scored in one request.",
        )

    try:
        results = predict_rows(dataset, run, payload.rows)
    except TrainingError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from exc
    except DatasetError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from exc

    task = (run.metrics or {}).get("task", "classification")
    return PredictResponse(
        task=task,
        predictions=[
            PredictionOut(
                index=row["index"],
                predicted=row["predicted"],
                confidence=row["confidence"],
                probabilities=row["probabilities"],
            )
            for row in results
        ],
    )


@router.get("/{run_id}/weights")
def get_weights(run_id: int, session: DbSession, user: CurrentUser) -> dict:
    """Expose the trained parameters as plain nested lists.

    This exists because a model you cannot inspect is a model you have to trust, and the
    whole premise here is that you should not have to.
    """
    run = _owned_run(session, user.id, run_id)
    if not run.weights_json:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail="This run has no stored weights."
        )
    blob = json.loads(run.weights_json)
    return {
        "run_id": run.id,
        "config": run.config,
        "shapes": [[len(w), len(w[0]) if w else 0] for w in blob["weights"]],
        "weights": blob["weights"],
        "biases": blob["biases"],
        "param_count": run.param_count,
        "note": "Rows are input units, columns are output units. Bias is a 1xN row.",
    }
