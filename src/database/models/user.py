"""ORM model for authentication — foundation only."""
from __future__ import annotations

import uuid

from sqlalchemy import Boolean, DateTime, JSON, String, func
from sqlalchemy.orm import Mapped, mapped_column

from src.database.base import Base


class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    created_at: Mapped[str] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[str] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    email:    Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    username: Mapped[str] = mapped_column(String(100), unique=True, nullable=False, index=True)

    # bcrypt hash — never expose in API responses
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)

    is_active:     Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    is_superuser:  Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    # viewer | analyst | admin
    role: Mapped[str] = mapped_column(String(50), nullable=False, default="viewer")

    # free | pro | enterprise
    subscription_tier: Mapped[str] = mapped_column(
        String(50), nullable=False, default="free"
    )

    last_login: Mapped[str | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    # Per-user preferences stored as a JSON dict
    preferences: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)

    def __repr__(self) -> str:
        return f"<User {self.username} role={self.role}>"
