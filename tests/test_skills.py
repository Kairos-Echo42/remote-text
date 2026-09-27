from __future__ import annotations

from agentforge.plugins.skills import SkillCatalog


def test_declarative_skill_catalog_loads_instructions(tmp_path):
    skill_dir = tmp_path / "demo"
    skill_dir.mkdir()
    (skill_dir / "skill.yaml").write_text(
        "name: demo-skill\nversion: 1.0.0\ndescription: Demo\n",
        encoding="utf-8",
    )
    (skill_dir / "SKILL.md").write_text("# Demo\n\nAlways cite evidence.", encoding="utf-8")
    catalog = SkillCatalog.load(tmp_path)
    assert catalog.get("demo-skill") is not None
    assert catalog.instructions_for(["demo-skill"])[0][1].startswith("# Demo")
