from __future__ import annotations

import asyncio
import os
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from agentforge.config import get_settings
from agentforge.logging import get_logger

logger = get_logger(__name__)


@dataclass(slots=True)
class SandboxProcessResult:
    exit_code: int
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False
    metadata: dict[str, str] = field(default_factory=dict)


class Sandbox(Protocol):
    workdir: str

    async def execute(
        self, command: str, *, timeout_seconds: int = 120, cwd: str | None = None
    ) -> SandboxProcessResult: ...

    async def read_file(self, path: str) -> str: ...

    async def write_file(self, path: str, content: str) -> None: ...

    async def list_dir(self, path: str = ".") -> list[dict[str, object]]: ...

    async def close(self) -> None: ...


class LocalSandbox:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.workdir = str(self.root)

    def _resolve(self, relative_path: str) -> Path:
        target = (self.root / relative_path).resolve()
        if self.root not in target.parents and target != self.root:
            raise ValueError("path escapes sandbox root")
        return target

    async def execute(
        self, command: str, *, timeout_seconds: int = 120, cwd: str | None = None
    ) -> SandboxProcessResult:
        working_directory = self._resolve(cwd or ".")
        if os.name == "nt":
            process = await asyncio.create_subprocess_shell(
                command,
                cwd=working_directory,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        else:
            process = await asyncio.create_subprocess_shell(
                command,
                cwd=working_directory,
                executable="/bin/sh",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout_seconds)
            return SandboxProcessResult(
                exit_code=process.returncode or 0,
                stdout=stdout.decode("utf-8", errors="replace"),
                stderr=stderr.decode("utf-8", errors="replace"),
            )
        except TimeoutError:
            process.kill()
            await process.wait()
            return SandboxProcessResult(exit_code=-1, timed_out=True, stderr="command timed out")

    async def read_file(self, path: str) -> str:
        return self._resolve(path).read_text(encoding="utf-8", errors="replace")

    async def write_file(self, path: str, content: str) -> None:
        target = self._resolve(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")

    async def list_dir(self, path: str = ".") -> list[dict[str, object]]:
        target = self._resolve(path)
        if not target.exists():
            return []
        return [
            {
                "name": item.name,
                "path": str(item.relative_to(self.root)).replace("\\", "/"),
                "is_dir": item.is_dir(),
                "size": item.stat().st_size if item.is_file() else 0,
            }
            for item in sorted(target.iterdir(), key=lambda item: item.name)
        ]

    async def close(self) -> None:
        return None


class DockerSandbox:
    def __init__(self, container, root: Path):
        self.container = container
        self.root = root.resolve()
        self.workdir = "/workspace"
        self._closed = False

    def _container_path(self, path: str) -> str:
        normalized = path.replace("\\", "/").lstrip("/")
        if ".." in Path(normalized).parts:
            raise ValueError("path escapes sandbox root")
        return f"/workspace/{normalized}" if normalized else "/workspace"

    async def execute(
        self, command: str, *, timeout_seconds: int = 120, cwd: str | None = None
    ) -> SandboxProcessResult:
        path = self._container_path(cwd or ".")
        try:
            result = await asyncio.wait_for(
                asyncio.to_thread(
                    self.container.exec_run,
                    ["sh", "-lc", command],
                    workdir=path,
                    stdout=True,
                    stderr=True,
                ),
                timeout=timeout_seconds,
            )
            output = result.output.decode("utf-8", errors="replace") if result.output else ""
            return SandboxProcessResult(exit_code=result.exit_code, stdout=output)
        except TimeoutError:
            try:
                await asyncio.to_thread(self.container.kill)
            except Exception:
                pass
            return SandboxProcessResult(exit_code=-1, timed_out=True, stderr="command timed out")

    async def read_file(self, path: str) -> str:
        target = self.root / path
        return target.read_text(encoding="utf-8", errors="replace")

    async def write_file(self, path: str, content: str) -> None:
        target = self.root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")

    async def list_dir(self, path: str = ".") -> list[dict[str, object]]:
        return await LocalSandbox(self.root).list_dir(path)

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            await asyncio.to_thread(self.container.remove, force=True)
        except Exception:
            pass


class SandboxProvider:
    def __init__(self, *, provider: str | None = None):
        settings = get_settings()
        self.provider = provider or ("local" if settings.is_test else "docker")
        self.root = settings.workspace_root
        self.host_root = settings.host_workspace_root
        self.image = settings.docker_image
        self.network_enabled = settings.docker_network_enabled
        self._client = None

    def _docker_client(self):
        if self._client is None:
            import docker

            self._client = docker.from_env()
        return self._client

    async def create(self, *, run_id: uuid.UUID, node_run_id: uuid.UUID, attempt: int) -> Sandbox:
        root = self.root / str(run_id) / str(node_run_id) / str(attempt)
        root.mkdir(parents=True, exist_ok=True)
        if self.provider == "local":
            return LocalSandbox(root)
        host_root = (self.host_root or str(self.root.resolve())).rstrip("/\\")
        docker_root = f"{host_root}/{run_id}/{node_run_id}/{attempt}"
        client = self._docker_client()
        container = await asyncio.to_thread(
            client.containers.run,
            self.image,
            ["sleep", "infinity"],
            detach=True,
            working_dir="/workspace",
            volumes={docker_root: {"bind": "/workspace", "mode": "rw"}},
            network_disabled=not self.network_enabled,
            read_only=True,
            cap_drop=["ALL"],
            security_opt=["no-new-privileges"],
            mem_limit="512m",
            nano_cpus=1_000_000_000,
            pids_limit=128,
            user="65534:65534",
            tmpfs={"/tmp": "rw,noexec,nosuid,size=64m"},
            labels={"agentforge.run_id": str(run_id), "agentforge.node_run_id": str(node_run_id)},
        )
        return DockerSandbox(container, root)
