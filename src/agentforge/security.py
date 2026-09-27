from __future__ import annotations

import base64
import hashlib
import hmac
import secrets
from datetime import UTC, datetime, timedelta
from typing import Any

import jwt
from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError
from cryptography.fernet import Fernet, InvalidToken

from agentforge.config import get_settings

_password_hasher = PasswordHasher(time_cost=3, memory_cost=65536, parallelism=4)


class AuthenticationError(ValueError):
    pass


class SecretDecryptionError(ValueError):
    pass


def hash_password(password: str) -> str:
    return _password_hasher.hash(password)


def verify_password(password: str, password_hash: str) -> bool:
    try:
        return _password_hasher.verify(password_hash, password)
    except VerifyMismatchError:
        return False


def create_access_token(
    subject: str,
    *,
    expires_delta: timedelta = timedelta(hours=12),
    extra: dict[str, Any] | None = None,
) -> str:
    now = datetime.now(UTC)
    payload: dict[str, Any] = {
        "sub": subject,
        "iat": now,
        "exp": now + expires_delta,
        "type": "access",
    }
    if extra:
        payload.update(extra)
    return jwt.encode(payload, get_settings().secret_key, algorithm="HS256")


def decode_access_token(token: str) -> dict[str, Any]:
    try:
        payload = jwt.decode(token, get_settings().secret_key, algorithms=["HS256"])
    except jwt.PyJWTError as exc:
        raise AuthenticationError("invalid or expired session") from exc
    if payload.get("type") != "access":
        raise AuthenticationError("invalid token type")
    return payload


def generate_api_key() -> tuple[str, str, str]:
    raw = f"af_{secrets.token_urlsafe(32)}"
    prefix = raw[:12]
    return raw, prefix, hash_api_key(raw)


def hash_api_key(raw: str) -> str:
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def verify_api_key(raw: str, expected_hash: str) -> bool:
    return hmac.compare_digest(hash_api_key(raw), expected_hash)


def generate_csrf_token() -> str:
    return secrets.token_urlsafe(32)


def _fernet() -> Fernet:
    configured = get_settings().master_key
    try:
        if len(configured) == 44:
            return Fernet(configured.encode("ascii"))
        digest = hashlib.sha256(configured.encode("utf-8")).digest()
        return Fernet(base64.urlsafe_b64encode(digest))
    except (ValueError, TypeError) as exc:
        raise SecretDecryptionError("invalid AGENTFORGE_MASTER_KEY") from exc


def encrypt_secret(value: str) -> str:
    return _fernet().encrypt(value.encode("utf-8")).decode("ascii")


def decrypt_secret(value: str) -> str:
    try:
        return _fernet().decrypt(value.encode("ascii")).decode("utf-8")
    except (InvalidToken, ValueError) as exc:
        raise SecretDecryptionError("unable to decrypt workspace secret") from exc


def redact(value: Any) -> Any:
    secret_keys = {"password", "token", "api_key", "apikey", "authorization", "secret"}
    if isinstance(value, dict):
        return {key: "***REDACTED***" if key.lower() in secret_keys else redact(item) for key, item in value.items()}
    if isinstance(value, list):
        return [redact(item) for item in value]
    return value
