from __future__ import annotations


def test_training_dataset_and_experiment_api(authenticated_client):
    workspace = authenticated_client.get("/api/v1/workspaces").json()[0]
    workspace_id = workspace["id"]

    recipes = authenticated_client.get(f"/api/v1/workspaces/{workspace_id}/training/recipes")
    assert recipes.status_code == 200
    assert any(
        item["recipe_id"] == "baseline.tabular.classification.dummy@1.0.0"
        for item in recipes.json()["recipes"]
    )
    baselines = authenticated_client.get(f"/api/v1/workspaces/{workspace_id}/training/baselines")
    assert baselines.status_code == 200
    assert any(item["strategy_id"] == "baseline.tabular.classification.dummy.prior" for item in baselines.json())

    dataset = authenticated_client.post(
        f"/api/v1/workspaces/{workspace_id}/datasets",
        json={
            "name": "api-training-dataset",
            "kind": "tabular",
            "task_type": "tabular_classification",
        },
    )
    assert dataset.status_code == 201
    dataset_id = dataset.json()["id"]

    csv_content = "feature,target\n" + "\n".join(
        f"{index % 5},{index % 2}" for index in range(40)
    )
    uploaded = authenticated_client.post(
        f"/api/v1/datasets/{dataset_id}/versions?target_column=target&split_seed=11",
        files={"file": ("training.csv", csv_content, "text/csv")},
    )
    assert uploaded.status_code == 201
    version = uploaded.json()
    assert version["status"] == "ready"
    assert version["split_checksum"]
    assert version["split_algorithm"] == "stratified"

    experiment = authenticated_client.post(
        f"/api/v1/workspaces/{workspace_id}/experiments",
        json={
            "name": "api-experiment",
            "dataset_version_id": version["id"],
            "budget": {
                "max_jobs": 2,
                "max_rounds": 1,
                "max_job_seconds": 30,
                "max_total_seconds": 60,
                "max_gpu_seconds": 0,
            },
            "selection_policy": {
                "validation_metric": "f1_macro",
                "direction": "maximize",
                "minimum_improvement": 0.01,
            },
        },
    )
    assert experiment.status_code == 201
    payload = experiment.json()
    assert payload["status"] == "running"
    assert payload["baseline_job_id"]
    assert payload["reserved_total_seconds"] == 30

    jobs = authenticated_client.get(f"/api/v1/experiments/{payload['id']}/jobs")
    assert jobs.status_code == 200
    assert len(jobs.json()) == 1
    assert jobs.json()[0]["job_kind"] == "baseline"

    report = authenticated_client.get(f"/api/v1/experiments/{payload['id']}/report")
    assert report.status_code == 200
    assert report.json()["budget_ledger"]["reserved_total_seconds"] == 30
    assert report.json()["final_test_visibility"] == "human_review_only"

    reservations = authenticated_client.get(f"/api/v1/experiments/{payload['id']}/budget-reservations")
    assert reservations.status_code == 200
    assert len(reservations.json()) == 1
    assert reservations.json()[0]["status"] == "active"

    invalid_job = authenticated_client.post(
        f"/api/v1/experiments/{payload['id']}/jobs",
        json={
            "experiment_id": payload["id"],
            "round_number": 1,
            "jobs": [
                {
                    "recipe_id": "tabular.classification.logistic@1.0.0",
                    "dataset_version_id": version["id"],
                    "parameters": {"C": 999999},
                    "device_policy": "cpu_only",
                    "max_job_seconds": 30,
                }
            ],
        },
    )
    assert invalid_job.status_code == 409
    assert invalid_job.json()["reason_code"] == "recipe_parameter_invalid"

    default_budget_experiment = authenticated_client.post(
        f"/api/v1/workspaces/{workspace_id}/experiments",
        json={
            "name": "api-default-budget",
            "dataset_version_id": version["id"],
        },
    )
    assert default_budget_experiment.status_code == 201
    assert default_budget_experiment.json()["reserved_total_seconds"] == 300


def test_training_web_pages_render(authenticated_client):
    for path in ("/app/datasets", "/app/recipes", "/app/experiments", "/app/training", "/app/model-registry"):
        response = authenticated_client.get(path)
        assert response.status_code == 200
        assert "AgentForge" in response.text
