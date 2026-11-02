"""Dataset upload, listing, preview and deletion."""

from __future__ import annotations

import csv
import io
import json

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile, status

from app.api.deps import AppSettings, CurrentUser, DbSession
from app.ml.dataset import DatasetError, parse_csv
from app.models import Dataset, TrainingRun
from app.schemas import ColumnOut, DatasetOut, DatasetPreview

router = APIRouter(prefix="/api/datasets", tags=["datasets"])


def _to_out(dataset: Dataset) -> DatasetOut:
    return DatasetOut(
        id=dataset.id,
        name=dataset.name,
        target=dataset.target,
        task=dataset.task,
        fingerprint=dataset.fingerprint,
        row_count=dataset.row_count,
        column_count=dataset.column_count,
        created_at=dataset.created_at.isoformat(),
        profile=[ColumnOut(**column) for column in dataset.profile],
    )


def _owned(session: DbSession, user_id: int, dataset_id: int) -> Dataset:
    dataset = session.get(Dataset, dataset_id)
    # 404 rather than 403 for someone else's dataset: telling a stranger that a record
    # exists is itself a leak.
    if dataset is None or dataset.owner_id != user_id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Dataset not found.")
    return dataset


@router.post("", response_model=DatasetOut, status_code=status.HTTP_201_CREATED)
async def upload_dataset(
    session: DbSession,
    user: CurrentUser,
    settings: AppSettings,
    request: Request,
    file: UploadFile = File(...),
    target: str = Form(...),
    name: str = Form(""),
) -> DatasetOut:
    limiter = getattr(request.app.state, "limiter", None)
    if limiter is not None:
        limiter.check(request, settings.rate_limit_write)

    raw = await file.read()
    if len(raw) > settings.max_upload_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=(
                f"File is {len(raw) / 1_000_000:.1f} MB; the limit is "
                f"{settings.max_upload_bytes / 1_000_000:.0f} MB."
            ),
        )
    if not raw:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="The uploaded file is empty."
        )

    try:
        parsed = parse_csv(
            raw,
            target=target.strip(),
            max_rows=settings.max_dataset_rows,
            max_columns=settings.max_dataset_columns,
        )
    except DatasetError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from exc

    dataset = Dataset(
        owner_id=user.id,
        name=(name.strip() or file.filename or "Untitled dataset")[:200],
        target=parsed.target,
        task=parsed.task,
        fingerprint=parsed.fingerprint,
        row_count=len(parsed.rows),
        column_count=len(parsed.columns),
        csv_body=raw.decode("utf-8-sig"),
        profile_json=json.dumps([p.as_dict() for p in parsed.profiles]),
    )
    session.add(dataset)
    session.commit()
    session.refresh(dataset)
    return _to_out(dataset)


@router.get("", response_model=list[DatasetOut])
def list_datasets(session: DbSession, user: CurrentUser) -> list[DatasetOut]:
    rows = (
        session.query(Dataset)
        .filter(Dataset.owner_id == user.id)
        .order_by(Dataset.created_at.desc())
        .limit(100)
        .all()
    )
    return [_to_out(d) for d in rows]


@router.get("/{dataset_id}", response_model=DatasetOut)
def get_dataset(dataset_id: int, session: DbSession, user: CurrentUser) -> DatasetOut:
    return _to_out(_owned(session, user.id, dataset_id))


@router.get("/{dataset_id}/preview", response_model=DatasetPreview)
def preview_dataset(
    dataset_id: int, session: DbSession, user: CurrentUser, limit: int = 10
) -> DatasetPreview:
    dataset = _owned(session, user.id, dataset_id)
    rows = list(csv.reader(io.StringIO(dataset.csv_body)))
    if not rows:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Empty dataset."
        )
    header = rows[0]
    body = rows[1 : max(1, min(limit, 50)) + 1]
    return DatasetPreview(columns=header, rows=body)


@router.delete("/{dataset_id}", status_code=status.HTTP_204_NO_CONTENT, response_model=None)
def delete_dataset(dataset_id: int, session: DbSession, user: CurrentUser):
    dataset = _owned(session, user.id, dataset_id)
    session.query(TrainingRun).filter(TrainingRun.dataset_id == dataset.id).delete(
        synchronize_session=False
    )
    session.delete(dataset)
    session.commit()
