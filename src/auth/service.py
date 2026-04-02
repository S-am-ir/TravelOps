"""
Auth service — register, login, JWT token management.
Uses bcrypt for password hashing and python-jose for JWT.
Per-user email config stored as SMTP credentials (app password).
"""

import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional
from jose import JWTError, jwt
import bcrypt
import asyncpg
from pydantic import BaseModel, Field

from src.config.settings import settings


# Password hashing
def _hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def _verify_password(plain: str, hashed: str) -> bool:
    return bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))


# JWT config
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 7  # 7 days


class UserProfile(BaseModel):
    id: str
    email: str
    email_configured: bool = False
    email_address: Optional[str] = None
    active_thread_id: Optional[str] = None
    created_at: Optional[str] = None


class RegisterRequest(BaseModel):
    email: str
    password: str = Field(..., min_length=6)


class LoginRequest(BaseModel):
    email: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserProfile


def create_access_token(user_id: str, email: str) -> str:
    expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    payload = {"sub": user_id, "email": email, "exp": expire}
    secret = (
        settings.jwt_secret.get_secret_value()
        if settings.jwt_secret
        else "dev-secret-change-me"
    )
    return jwt.encode(payload, secret, algorithm=ALGORITHM)


def decode_access_token(token: str) -> Optional[dict]:
    try:
        secret = (
            settings.jwt_secret.get_secret_value()
            if settings.jwt_secret
            else "dev-secret-change-me"
        )
        payload = jwt.decode(token, secret, algorithms=[ALGORITHM])
        return payload
    except JWTError:
        return None


async def _get_pool() -> asyncpg.Pool:
    global _pool
    if _pool is None:
        conn_string = (
            settings.supabase_url.get_secret_value() if settings.supabase_url else None
        )
        if not conn_string:
            raise RuntimeError(
                "No database connection configured (SUPABASE_URL not set)"
            )
        _pool = await asyncpg.create_pool(conn_string, min_size=1, max_size=5)
    return _pool


_pool: Optional[asyncpg.Pool] = None


async def init_auth_db():
    pool = await _get_pool()
    async with pool.acquire() as conn:
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS user_profiles (
                id              TEXT PRIMARY KEY DEFAULT gen_random_uuid()::text,
                email           TEXT UNIQUE NOT NULL,
                password_hash   TEXT NOT NULL,
                smtp_email      TEXT,
                smtp_password   TEXT,
                active_thread_id TEXT,
                created_at      TIMESTAMPTZ DEFAULT now()
            );
        """)
        # Migrations: add new columns, drop old ones
        for col in ["smtp_email", "smtp_password"]:
            try:
                await conn.execute(
                    f"ALTER TABLE user_profiles ADD COLUMN IF NOT EXISTS {col} TEXT"
                )
            except Exception:
                pass
        for col in [
            "gmail_access_token",
            "gmail_refresh_token",
            "gmail_email",
            "telegram_bot_token_enc",
            "telegram_chat_id",
        ]:
            try:
                await conn.execute(
                    f"ALTER TABLE user_profiles DROP COLUMN IF EXISTS {col}"
                )
            except Exception:
                pass
    print("[Auth] Database tables ready")


async def close_auth_db():
    global _pool
    if _pool:
        await _pool.close()
        _pool = None


class AuthService:
    @staticmethod
    async def register(email: str, password: str) -> TokenResponse:
        pool = await _get_pool()
        async with pool.acquire() as conn:
            existing = await conn.fetchrow(
                "SELECT id FROM user_profiles WHERE email = $1", email.lower()
            )
            if existing:
                raise ValueError("Email already registered")
            user_id = str(uuid.uuid4())
            hashed = _hash_password(password)
            await conn.execute(
                "INSERT INTO user_profiles (id, email, password_hash) VALUES ($1, $2, $3)",
                user_id,
                email.lower(),
                hashed,
            )
        token = create_access_token(user_id, email.lower())
        return TokenResponse(
            access_token=token,
            user=UserProfile(id=user_id, email=email.lower()),
        )

    @staticmethod
    async def login(email: str, password: str) -> TokenResponse:
        pool = await _get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT id, email, password_hash, smtp_email, active_thread_id, created_at::text "
                "FROM user_profiles WHERE email = $1",
                email.lower(),
            )
        if not row:
            raise ValueError("Invalid email or password")
        if not _verify_password(password, row["password_hash"]):
            raise ValueError("Invalid email or password")
        user = UserProfile(
            id=row["id"],
            email=row["email"],
            email_configured=bool(row["smtp_email"]),
            email_address=row["smtp_email"],
            active_thread_id=row["active_thread_id"],
            created_at=row["created_at"],
        )
        token = create_access_token(row["id"], row["email"])
        return TokenResponse(access_token=token, user=user)

    @staticmethod
    async def get_user(user_id: str) -> Optional[UserProfile]:
        pool = await _get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT id, email, smtp_email, active_thread_id, created_at::text "
                "FROM user_profiles WHERE id = $1",
                user_id,
            )
        if not row:
            return None
        return UserProfile(
            id=row["id"],
            email=row["email"],
            email_configured=bool(row["smtp_email"]),
            email_address=row["smtp_email"],
            active_thread_id=row["active_thread_id"],
            created_at=row["created_at"],
        )

    @staticmethod
    async def save_smtp_config(
        user_id: str, smtp_email: str, smtp_password: str
    ) -> bool:
        pool = await _get_pool()
        async with pool.acquire() as conn:
            result = await conn.execute(
                "UPDATE user_profiles SET smtp_email = $2, smtp_password = $3 WHERE id = $1",
                user_id,
                smtp_email,
                smtp_password,
            )
        return result == "UPDATE 1"

    @staticmethod
    async def get_smtp_config(user_id: str) -> Optional[dict]:
        pool = await _get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT smtp_email, smtp_password FROM user_profiles WHERE id = $1",
                user_id,
            )
        if not row or not row["smtp_email"]:
            return None
        return {
            "smtp_email": row["smtp_email"],
            "smtp_password": row["smtp_password"],
        }

    @staticmethod
    async def clear_smtp_config(user_id: str):
        pool = await _get_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                "UPDATE user_profiles SET smtp_email = NULL, smtp_password = NULL WHERE id = $1",
                user_id,
            )

    @staticmethod
    async def update_active_thread(user_id: str, thread_id: str):
        pool = await _get_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                "UPDATE user_profiles SET active_thread_id = $2 WHERE id = $1",
                user_id,
                thread_id,
            )

    @staticmethod
    async def get_active_thread(user_id: str) -> Optional[str]:
        pool = await _get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT active_thread_id FROM user_profiles WHERE id = $1",
                user_id,
            )
        return row["active_thread_id"] if row else None

    @staticmethod
    async def clear_active_thread(user_id: str):
        pool = await _get_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                "UPDATE user_profiles SET active_thread_id = NULL WHERE id = $1",
                user_id,
            )


async def get_auth_service() -> AuthService:
    return AuthService()
