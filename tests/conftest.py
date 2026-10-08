# ruff: noqa: E402
from __future__ import annotations

import os
import tempfile
from pathlib import Path

TEST_ROOT = Path(tempfile.gettempdir()) / "agentforge-tests"
TEST_ROOT.mkdir(parents=True, exist_ok=True)
(TEST_ROOT / "artifacts").mkdir(parents=True, exist_ok=True)
(TEST_ROOT / "workspaces").mkdir(parents=True, exist_ok=True)
(TEST_ROOT / "training").mkdir(parents=True, exist_ok=True)
TEST_DATABASE = (TEST_ROOT / "agentforge-test.db").as_posix()

os.environ.update(
    {
        "AGENTFORGE_ENV": "test",
        "AGENTFORGE_DATABASE_URL": f"sqlite+aiosqlite:///{TEST_DATABASE}",
        "AGENTFORGE_REDIS_URL": "redis://localhost:6399/15",
        "AGENTFORGE_ARTIFACT_ROOT": str(TEST_ROOT / "artifacts"),
        "AGENTFORGE_WORKSPACE_ROOT": str(TEST_ROOT / "workspaces"),
        "AGENTFORGE_TRAINING_ROOT": str(TEST_ROOT / "training"),
        "AGENTFORGE_HOST_TRAINING_ROOT": str(TEST_ROOT / "training"),
        "AGENTFORGE_SECRET_KEY": "test-secret-key-at-least-32-bytes-long",
        "AGENTFORGE_MASTER_KEY": "dGVzdC1tYXN0ZXIta2V5LTMyLWJ5dGVzISE=",
        "AGENTFORGE_ADMIN_EMAIL": "admin@example.com",
        "AGENTFORGE_ADMIN_PASSWORD": "test-password-123",
        "DEEPSEEK_API_KEY": "",
        "DASHSCOPE_API_KEY": "",
    }
)

import pytest
import pytest_asyncio
from fastapi.testclient import TestClient

from agentforge.config import get_settings, reset_settings_cache
from agentforge.db import create_all, drop_all, reset_database_state
from agentforge.gateway.app import create_app
from agentforge.services.auth import bootstrap_default_admin


@pytest.fixture(scope="session", autouse=True)
def _test_environment():
    reset_settings_cache()
    reset_database_state()
    get_settings().ensure_directories()
    yield
    reset_database_state()
    reset_settings_cache()


@pytest_asyncio.fixture(autouse=True)
async def reset_database(_test_environment):
    reset_database_state()
    await drop_all()
    await create_all()
    await bootstrap_default_admin()
    yield
    await drop_all()


@pytest.fixture()
def client(_test_environment):
    reset_database_state()
    with TestClient(create_app()) as test_client:
        yield test_client


@pytest.fixture()
def authenticated_client(client: TestClient):
    response = client.post(
        "/login",
        data={"email": "admin@example.com", "password": "test-password-123"},
        follow_redirects=False,
    )
    assert response.status_code == 303
    return client
