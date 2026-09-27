from __future__ import annotations

from agentforge.security import (
    decrypt_secret,
    encrypt_secret,
    generate_api_key,
    hash_password,
    verify_api_key,
    verify_password,
)


def test_password_api_key_and_secret_roundtrips():
    password_hash = hash_password("correct horse battery staple")
    assert verify_password("correct horse battery staple", password_hash)
    assert not verify_password("wrong", password_hash)

    raw, prefix, digest = generate_api_key()
    assert raw.startswith("af_")
    assert prefix == raw[:12]
    assert verify_api_key(raw, digest)

    encrypted = encrypt_secret("workspace-secret")
    assert encrypted != "workspace-secret"
    assert decrypt_secret(encrypted) == "workspace-secret"
