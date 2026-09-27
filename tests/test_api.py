from __future__ import annotations


def test_health_and_authenticated_web_pages(authenticated_client):
    health = authenticated_client.get("/health/live")
    assert health.status_code == 200
    assert health.json() == {"status": "ok"}

    dashboard = authenticated_client.get("/app")
    assert dashboard.status_code == 200
    assert "AgentForge" in dashboard.text

    workbench = authenticated_client.get("/app/workbench")
    assert workbench.status_code == 200
    assert "任务工作台" in workbench.text


def test_workspace_api_uses_session_cookie(authenticated_client):
    response = authenticated_client.get("/api/v1/workspaces")
    assert response.status_code == 200
    payload = response.json()
    assert len(payload) == 1
    assert payload[0]["slug"] == "default"


def test_unauthenticated_api_is_rejected(client):
    response = client.get("/api/v1/workspaces")
    assert response.status_code == 401
