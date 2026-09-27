# Contributing

## Development Flow

1. Create the `agentforge` Conda environment with Python 3.12.
2. Install `requirements-dev.txt` and the project with `pip install -e . --no-deps`.
3. Start PostgreSQL and Redis with Docker Compose.
4. Run migrations with `alembic upgrade head`.
5. Add or update tests before changing runtime behavior.
6. Run `scripts/check.ps1`.

## Engineering Rules

- Keep the outer DAG and inner Agent runtime boundaries intact.
- Add new executable tools through the Capability interface.
- Persist authoritative state in PostgreSQL; Redis is transport.
- Never log or return plaintext workspace secrets.
- New side-effectful capabilities must declare idempotency and side-effect level.
- Workflow and Agent YAML changes require validation tests.