"""Registration, login and current-user endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request, status
from sqlalchemy.exc import IntegrityError

from app.api.deps import AppSettings, CurrentUser, DbSession
from app.core.security import create_access_token, hash_password, verify_password
from app.models import User
from app.schemas import LoginRequest, RegisterRequest, TokenResponse, UserOut

router = APIRouter(prefix="/api/auth", tags=["auth"])


def _issue(user: User, settings: AppSettings) -> TokenResponse:
    token = create_access_token(str(user.id), settings)
    return TokenResponse(
        access_token=token,
        user=UserOut(id=user.id, email=user.email, display_name=user.display_name),
    )


@router.post("/register", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
def register(
    payload: RegisterRequest,
    session: DbSession,
    settings: AppSettings,
    request: Request,
) -> TokenResponse:
    limiter = getattr(request.app.state, "limiter", None)
    if limiter is not None:
        limiter.check(request, settings.rate_limit_auth)

    email = payload.email.lower()
    if session.query(User).filter(User.email == email).first():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="An account with that email already exists.",
        )

    user = User(
        email=email,
        password_hash=hash_password(payload.password),
        display_name=payload.display_name.strip(),
    )
    session.add(user)
    try:
        session.commit()
    except IntegrityError as exc:  # lost a race to the unique index
        session.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="An account with that email already exists.",
        ) from exc
    session.refresh(user)
    return _issue(user, settings)


@router.post("/login", response_model=TokenResponse)
def login(
    payload: LoginRequest,
    session: DbSession,
    settings: AppSettings,
    request: Request,
) -> TokenResponse:
    limiter = getattr(request.app.state, "limiter", None)
    if limiter is not None:
        limiter.check(request, settings.rate_limit_auth)

    user = session.query(User).filter(User.email == payload.email.lower()).first()
    # Same message and same work whether the email exists or not, so timing and wording
    # do not leak account existence.
    if user is None or not verify_password(payload.password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Email or password is incorrect.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return _issue(user, settings)


@router.get("/me", response_model=UserOut)
def me(user: CurrentUser) -> UserOut:
    return UserOut(id=user.id, email=user.email, display_name=user.display_name)
