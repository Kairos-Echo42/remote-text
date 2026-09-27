from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class SkillManifest(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    version: str = "1.0.0"
    description: str | None = None
    instructions_file: str = "SKILL.md"
    entry_point: str | None = None
    capabilities: list[str] = Field(default_factory=list)


@dataclass(slots=True)
class Skill:
    manifest: SkillManifest
    path: Path
    instructions: str


@dataclass(slots=True)
class SkillCatalog:
    skills: dict[str, Skill] = field(default_factory=dict)

    @classmethod
    def load(cls, root: str | Path) -> SkillCatalog:
        catalog = cls()
        base = Path(root)
        if not base.exists():
            return catalog
        for manifest_path in base.glob("*/skill.yaml"):
            payload = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
            manifest = SkillManifest.model_validate(payload)
            instructions_path = manifest_path.parent / manifest.instructions_file
            instructions = (
                instructions_path.read_text(encoding="utf-8")
                if instructions_path.exists()
                else manifest.description or ""
            )
            catalog.skills[manifest.name] = Skill(
                manifest=manifest,
                path=manifest_path.parent,
                instructions=instructions,
            )
        return catalog

    def get(self, name: str) -> Skill | None:
        return self.skills.get(name)

    def instructions_for(self, names: list[str]) -> list[tuple[str, str]]:
        result: list[tuple[str, str]] = []
        for name in names:
            skill = self.get(name)
            if skill:
                result.append((name, skill.instructions))
        return result

    def register_code_skills(self, registry) -> None:
        for skill in self.skills.values():
            if not skill.manifest.entry_point:
                continue
            module_name, _, attribute = skill.manifest.entry_point.partition(":")
            module = importlib.import_module(module_name)
            register = getattr(module, attribute or "register")
            register(registry)
