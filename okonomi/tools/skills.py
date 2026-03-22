"""Filesystem-based skills system for Okonomi.

Discovers and loads reusable workflow guides defined as markdown files
with YAML frontmatter. Scans both bundled skills (okonomi/skills/) and
user skills (~/.okonomi/skills/).

Supports conditional activation via frontmatter metadata:
  metadata:
    okonomi:
      requires_tools: [research]       # only show when these tools are loaded
      fallback_for_tools: [mcp_search] # hide when the primary tool IS available
      platforms: [macos, linux]        # restrict to specific OS platforms
"""

import logging
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path

import yaml

from okonomi.tools._metadata import ToolMetadataMixin

logger = logging.getLogger(__name__)

BUNDLED_SKILLS_DIR = Path(__file__).parent.parent / "skills"
USER_SKILLS_DIR = Path(os.getenv("OKONOMI_HOME", Path.home() / ".okonomi")) / "skills"

MAX_NAME_LENGTH = 64
MAX_DESCRIPTION_LENGTH = 1024
VALID_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
ALLOWED_SUBDIRS = {"references", "templates", "scripts", "assets"}

_PLATFORM_MAP = {
    "macos": "darwin",
    "linux": "linux",
    "windows": "win32",
}


def _parse_frontmatter(content: str) -> tuple[dict, str]:
    """Split a markdown file into YAML frontmatter and body."""
    if not content.startswith("---"):
        return {}, content

    parts = content.split("---", 2)
    if len(parts) < 3:
        return {}, content

    metadata = yaml.safe_load(parts[1]) or {}
    body = parts[2].strip()
    return metadata, body


def _is_platform_compatible(frontmatter: dict) -> bool:
    """Check if a skill is compatible with the current OS platform."""
    okonomi_meta = frontmatter.get("metadata", {}).get("okonomi", {})
    platforms = okonomi_meta.get("platforms") or frontmatter.get("platforms")
    if not platforms:
        return True
    current = sys.platform
    return any(current.startswith(_PLATFORM_MAP.get(p, p)) for p in platforms if _PLATFORM_MAP.get(p, p))


def _read_conditions(frontmatter: dict) -> dict:
    """Extract conditional activation fields from frontmatter."""
    okonomi = frontmatter.get("metadata", {}).get("okonomi", {})
    return {
        "requires_tools": okonomi.get("requires_tools", []),
        "fallback_for_tools": okonomi.get("fallback_for_tools", []),
    }


def _should_show(
    conditions: dict,
    available_tools: set[str] | None,
) -> bool:
    """Return False if a skill's conditional activation rules exclude it."""
    if available_tools is None:
        return True

    for t in conditions.get("fallback_for_tools", []):
        if t in available_tools:
            return False

    for t in conditions.get("requires_tools", []):
        if t not in available_tools:
            return False

    return True


def _discover_skills(
    skills_dirs: list[Path],
    available_tools: set[str] | None = None,
) -> dict[str, dict]:
    """Discover skills from one or more directories.

    Scans bundled and user skills dirs. User skills take precedence
    over bundled skills with the same name.
    """
    skills: dict[str, dict] = {}

    for skills_dir in skills_dirs:
        if not skills_dir.is_dir():
            continue

        for path in sorted(skills_dir.rglob("SKILL.md")):
            try:
                content = path.read_text()
                metadata, _ = _parse_frontmatter(content)
                name = metadata.get("name") or path.parent.stem
                if name in skills and skills_dir == BUNDLED_SKILLS_DIR:
                    continue
                if not _is_platform_compatible(metadata):
                    continue
                conditions = _read_conditions(metadata)
                if not _should_show(conditions, available_tools):
                    continue
                description = metadata.get("description", "")
                category = ""
                rel = path.relative_to(skills_dir)
                if len(rel.parts) > 2:
                    category = "/".join(rel.parts[:-2])
                skills[name] = {
                    "description": description,
                    "path": path,
                    "category": category,
                    "source": "user" if skills_dir == USER_SKILLS_DIR else "bundled",
                }
            except Exception:
                continue

        for path in sorted(skills_dir.glob("*.md")):
            try:
                content = path.read_text()
                metadata, _ = _parse_frontmatter(content)
                name = metadata.get("name") or path.stem
                if name in skills:
                    continue
                if not _is_platform_compatible(metadata):
                    continue
                conditions = _read_conditions(metadata)
                if not _should_show(conditions, available_tools):
                    continue
                description = metadata.get("description", "")
                skills[name] = {
                    "description": description,
                    "path": path,
                    "category": "",
                    "source": "user" if skills_dir == USER_SKILLS_DIR else "bundled",
                }
            except Exception:
                continue

    return skills


def _validate_name(name: str) -> str | None:
    if not name:
        return "Skill name is required."
    if len(name) > MAX_NAME_LENGTH:
        return f"Skill name exceeds {MAX_NAME_LENGTH} characters."
    if not VALID_NAME_RE.match(name):
        return (
            f"Invalid skill name '{name}'. Use lowercase letters, numbers, "
            f"hyphens, dots, and underscores. Must start with a letter or digit."
        )
    return None


def _validate_frontmatter_content(content: str) -> str | None:
    if not content.strip():
        return "Content cannot be empty."
    if not content.startswith("---"):
        return "SKILL.md must start with YAML frontmatter (---). See existing skills for format."
    end_match = re.search(r"\n---\s*\n", content[3:])
    if not end_match:
        return "SKILL.md frontmatter is not closed. Ensure you have a closing '---' line."
    yaml_content = content[3 : end_match.start() + 3]
    try:
        parsed = yaml.safe_load(yaml_content)
    except yaml.YAMLError as e:
        return f"YAML frontmatter parse error: {e}"
    if not isinstance(parsed, dict):
        return "Frontmatter must be a YAML mapping (key: value pairs)."
    if "name" not in parsed:
        return "Frontmatter must include 'name' field."
    if "description" not in parsed:
        return "Frontmatter must include 'description' field."
    if len(str(parsed["description"])) > MAX_DESCRIPTION_LENGTH:
        return f"Description exceeds {MAX_DESCRIPTION_LENGTH} characters."
    body = content[end_match.end() + 3 :].strip()
    if not body:
        return "SKILL.md must have content after the frontmatter."
    return None


def _atomic_write(file_path: Path, content: str) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_path = tempfile.mkstemp(
        dir=str(file_path.parent),
        prefix=".tmp.",
        suffix=f".{file_path.name}",
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
        os.replace(temp_path, file_path)
    except Exception:
        try:
            os.unlink(temp_path)
        except OSError:
            pass
        raise


def _validate_file_path(file_path: str) -> str | None:
    if not file_path:
        return "file_path is required."
    normalized = Path(file_path)
    if ".." in normalized.parts:
        return "Path traversal ('..') is not allowed."
    if not normalized.parts or normalized.parts[0] not in ALLOWED_SUBDIRS:
        allowed = ", ".join(sorted(ALLOWED_SUBDIRS))
        return f"File must be under one of: {allowed}. Got: '{file_path}'"
    if len(normalized.parts) < 2:
        return f"Provide a file path, not just a directory. Example: '{normalized.parts[0]}/myfile.md'"
    return None


class SkillTools(ToolMetadataMixin):
    """Tools for discovering, loading, and managing workflow skills."""

    __namespace__ = "skills"
    __description__ = "Tools for discovering, loading, and managing workflow skills"

    def __init__(self, config, available_tools: set[str] | None = None):
        self.config = config
        self._available_tools = available_tools
        self._skills_cache: dict[str, dict] | None = None

    def _get_skills(self) -> dict[str, dict]:
        if self._skills_cache is None:
            self._skills_cache = _discover_skills(
                [USER_SKILLS_DIR, BUNDLED_SKILLS_DIR],
                self._available_tools,
            )
        return self._skills_cache

    def _invalidate_cache(self) -> None:
        self._skills_cache = None

    def list_skills(self) -> dict:
        """List all available workflow skills.

        Returns:
            Dict with a "skills" key containing a list of skill summaries,
            each with "name", "description", "category", and "source".
        """
        skills = self._get_skills()
        return {
            "skills": [
                {
                    "name": name,
                    "description": info["description"],
                    "category": info.get("category", ""),
                    "source": info.get("source", "bundled"),
                }
                for name, info in skills.items()
            ]
        }

    def load_skill(self, name: str) -> dict:
        """Load the full instructions for a workflow skill.

        Args:
            name: Name of the skill to load.

        Returns:
            Dict with "name" and "instructions", or an error with available skill names.
        """
        skills = self._get_skills()
        if name not in skills:
            return {
                "error": f"Unknown skill '{name}'",
                "available_skills": list(skills.keys()),
            }

        content = skills[name]["path"].read_text()
        _, body = _parse_frontmatter(content)
        return {"name": name, "instructions": body}

    def create_skill(
        self,
        name: str,
        content: str,
        category: str = "",
    ) -> dict:
        """Create a new workflow skill in ~/.okonomi/skills/.

        The content must be a complete SKILL.md with YAML frontmatter containing
        at minimum 'name' and 'description' fields, followed by markdown instructions.

        Args:
            name: Skill name (lowercase, hyphens/underscores allowed).
            content: Full SKILL.md content including frontmatter.
            category: Optional category subdirectory.

        Returns:
            Dict with success status and path or error message.
        """
        err = _validate_name(name)
        if err:
            return {"success": False, "error": err}

        err = _validate_frontmatter_content(content)
        if err:
            return {"success": False, "error": err}

        existing = self._get_skills()
        if name in existing:
            return {
                "success": False,
                "error": f"A skill named '{name}' already exists at {existing[name]['path']}.",
            }

        skill_dir = USER_SKILLS_DIR / category / name if category else USER_SKILLS_DIR / name
        skill_dir.mkdir(parents=True, exist_ok=True)
        _atomic_write(skill_dir / "SKILL.md", content)

        self._invalidate_cache()
        return {
            "success": True,
            "message": f"Skill '{name}' created.",
            "path": str(skill_dir),
        }

    def edit_skill(self, name: str, content: str) -> dict:
        """Replace the SKILL.md content of an existing user skill.

        Args:
            name: Name of the skill to edit.
            content: New full SKILL.md content including frontmatter.

        Returns:
            Dict with success status or error message.
        """
        skills = self._get_skills()
        if name not in skills:
            return {"success": False, "error": f"Skill '{name}' not found.", "available_skills": list(skills.keys())}

        if skills[name].get("source") == "bundled":
            return {"success": False, "error": f"Cannot edit bundled skill '{name}'. Create a user copy instead."}

        err = _validate_frontmatter_content(content)
        if err:
            return {"success": False, "error": err}

        skill_path = skills[name]["path"]
        _atomic_write(skill_path, content)
        self._invalidate_cache()
        return {"success": True, "message": f"Skill '{name}' updated."}

    def patch_skill(self, name: str, old_text: str, new_text: str, file_path: str = "SKILL.md") -> dict:
        """Find-and-replace within a skill file.

        Args:
            name: Skill name.
            old_text: Text to find.
            new_text: Replacement text.
            file_path: File within the skill directory (default: SKILL.md).

        Returns:
            Dict with success status or error message.
        """
        skills = self._get_skills()
        if name not in skills:
            return {"success": False, "error": f"Skill '{name}' not found."}

        if skills[name].get("source") == "bundled":
            return {"success": False, "error": f"Cannot patch bundled skill '{name}'."}

        skill_dir = skills[name]["path"].parent
        target = skill_dir / file_path
        if not target.exists():
            return {"success": False, "error": f"File '{file_path}' not found in skill '{name}'."}

        content = target.read_text()
        if old_text not in content:
            return {"success": False, "error": "old_text not found in file."}

        new_content = content.replace(old_text, new_text, 1)
        if file_path == "SKILL.md":
            err = _validate_frontmatter_content(new_content)
            if err:
                return {"success": False, "error": err}

        _atomic_write(target, new_content)
        self._invalidate_cache()
        return {"success": True, "message": f"Patched '{file_path}' in skill '{name}'."}

    def delete_skill(self, name: str) -> dict:
        """Delete a user skill entirely.

        Args:
            name: Name of the skill to delete.

        Returns:
            Dict with success status or error message.
        """
        skills = self._get_skills()
        if name not in skills:
            return {"success": False, "error": f"Skill '{name}' not found."}

        if skills[name].get("source") == "bundled":
            return {"success": False, "error": f"Cannot delete bundled skill '{name}'."}

        skill_dir = skills[name]["path"].parent
        shutil.rmtree(skill_dir, ignore_errors=True)
        self._invalidate_cache()
        return {"success": True, "message": f"Skill '{name}' deleted."}

    def write_file(self, name: str, file_path: str, file_content: str) -> dict:
        """Add or overwrite a supporting file in a user skill.

        Files must be under one of: assets, references, scripts, templates.

        Args:
            name: Skill name.
            file_path: Relative path within skill dir (e.g. "references/api.md").
            file_content: Content to write.

        Returns:
            Dict with success status or error message.
        """
        skills = self._get_skills()
        if name not in skills:
            return {"success": False, "error": f"Skill '{name}' not found."}

        if skills[name].get("source") == "bundled":
            return {"success": False, "error": f"Cannot write files in bundled skill '{name}'."}

        err = _validate_file_path(file_path)
        if err:
            return {"success": False, "error": err}

        skill_dir = skills[name]["path"].parent
        target = skill_dir / file_path
        _atomic_write(target, file_content)
        return {"success": True, "message": f"Wrote '{file_path}' in skill '{name}'."}
