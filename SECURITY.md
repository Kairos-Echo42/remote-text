# Security

## Reporting

Report vulnerabilities privately to the project maintainer. Do not include secrets or production data in public issues.

## Default Controls

- Workspace-scoped repository access.
- Argon2 password hashing.
- HttpOnly JWT session cookies and CSRF checks.
- Hashed workspace API keys.
- Fernet-encrypted workspace secrets.
- Redacted event and log payloads.
- Docker Sandbox: non-root, read-only root, resource limits, no network by default.
- MCP stdio and Python Skills are trusted-code boundaries requiring workspace administrator access.

## Production Requirements

- Replace `AGENTFORGE_SECRET_KEY` and `AGENTFORGE_MASTER_KEY`.
- Protect Docker socket access; do not expose Workers as public endpoints.
- Run behind TLS and a trusted reverse proxy.
- Apply database backups, rotation, rate limits, and external secret management.