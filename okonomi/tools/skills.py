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

from okonomi.config import OKONOMI_SKILLS_DIR
from okonomi.tools._metadata import ToolMetadataMixin


logger = logging.getLogger(__name__)

BUNDLED_SKILLS_DIR = Path(__file__).parent.parent / "skills"
USER_SKILLS_DIR = OKONOMI_SKILLS_DIR

MAX_NAME_LENGTH = 64
MAX_DESCRIPTION_LENGTH = 1024
MAX_INTRO_CHARS = 2000
VALID_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
ALLOWED_SUBDIRS = {"references", "templates", "scripts", "assets"}

_PLATFORM_MAP = {
    "macos": "darwin",
    "linux": "linux",
    "windows": "win32",
}


_HEADING_RE = re.compile(r"^(#{2,3}) +(.+?)\s*$", re.MULTILINE)


def _slugify_heading(heading: str) -> str:
    """Stable slug from a heading. Lowercase, alphanumerics + hyphens."""
    s = re.sub(r"[^a-z0-9]+", "-", heading.lower()).strip("-")
    return s or "section"


def _parse_skill_sections(body: str) -> tuple[str, list[dict]]:
    """Split a skill body into (intro, sections).

    intro: text before the first ``## H2`` heading.
    sections: list of {slug, heading, content, level, parent_slug}.
        Level is 2 or 3. ``parent_slug`` is the slug of the enclosing
        H2 for an H3, or empty for H2s themselves.

    Slugs are flat and unique within a skill, so callers can look up
    any section by slug regardless of nesting depth. The parent_slug
    field lets the TOC present the hierarchy.

    Each section's ``content`` includes its own heading line and runs
    until the *next heading at any level* — so an H2's content covers
    only the prose between its heading and its first H3 child (the
    overview/intro for that group). The H3 children are separate
    entries with their own content.
    """
    matches = list(_HEADING_RE.finditer(body))
    if not matches:
        return body.strip(), []

    # First match must be H2 — anything before it is the intro.
    intro_end = next((m.start() for m in matches if len(m.group(1)) == 2), len(body))
    intro = body[:intro_end].strip()

    sections: list[dict] = []
    current_h2_slug = ""
    for i, m in enumerate(matches):
        level = len(m.group(1))
        heading = m.group(2).strip()
        slug = _slugify_heading(heading)
        end = matches[i + 1].start() if i + 1 < len(matches) else len(body)

        if level == 2:
            current_h2_slug = slug
            parent = ""
        else:  # level == 3
            parent = current_h2_slug

        sections.append(
            {
                "slug": slug,
                "heading": heading,
                "content": body[m.start() : end].rstrip(),
                "level": level,
                "parent_slug": parent,
            }
        )
    return intro, sections


def _list_skill_references(skill_dir: Path) -> list[str]:
    """Reference filenames (without `.md`) inside ``<skill>/references/``."""
    refs_dir = skill_dir / "references"
    if not refs_dir.is_dir():
        return []
    return sorted(p.stem for p in refs_dir.glob("*.md"))


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


def _check_skill_structure(content: str) -> tuple[str | None, list[str]]:
    """Validate that a SKILL.md discloses section by section.

    Returns ``(hard_error, warnings)``. ``hard_error`` is set only for
    problems that break ``load_skill_section`` — duplicate heading slugs,
    which make every colliding section past the first unreachable.
    ``warnings`` flag structure that still loads but not lazily: an intro
    (text before the first ``## H2``) large enough that ``load_skill``
    returns a heavy payload inline on every load.
    """
    _, body = _parse_frontmatter(content)
    intro, sections = _parse_skill_sections(body)

    slugs = [s["slug"] for s in sections]
    dupes = sorted({s for s in slugs if slugs.count(s) > 1})
    if dupes:
        joined = ", ".join(f"'{d}'" for d in dupes)
        return (
            f"Duplicate section slug(s) {joined}: multiple headings reduce to the "
            f"same slug, so load_skill_section can only reach the first and the rest "
            f"are unreachable. Rename the headings so each is distinct.",
            [],
        )

    warnings: list[str] = []
    if len(intro) > MAX_INTRO_CHARS:
        if not sections:
            warnings.append(
                f"Body has no '## ' (H2) sections and is {len(intro)} chars, so "
                f"load_skill returns the whole skill inline instead of disclosing it "
                f"section by section. Break the body into '## ' sections."
            )
        else:
            warnings.append(
                f"The text before the first '## ' heading is {len(intro)} chars; "
                f"load_skill returns it inline on every load. Keep it a short preamble "
                f"(under {MAX_INTRO_CHARS}) and move the bulk under H2 sections."
            )
    return None, warnings


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

    def __init__(self, available_tools: set[str] | None = None):
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

    def _load_full(self, name: str) -> dict:
        """Internal: full SKILL.md body. Used by core-skill inlining and
        slash-command paths that legitimately need every line."""
        skills = self._get_skills()
        if name not in skills:
            return {
                "error": f"Unknown skill '{name}'",
                "available_skills": list(skills.keys()),
            }
        content = skills[name]["path"].read_text()
        _, body = _parse_frontmatter(content)
        return {"name": name, "content": body}

    def load_skill(self, name: str) -> dict:
        """Load a skill's *table of contents* — intro + section list +
        reference list. Cheap; the heavy material stays on disk until
        you ask for a specific section or reference.

        Args:
            name: Name of the skill to load.

        Returns:
            Dict with:
              - ``description``: skill's frontmatter description
              - ``intro``: text before the first H2 heading (the "what
                is this skill" preamble — usually 5–15 lines)
              - ``sections``: list of ``{"slug", "heading"}`` for every
                ``## H2`` in the SKILL.md
              - ``references``: list of filenames (without ``.md``) in
                ``references/``
              - ``next``: prose hint on how to drill in

            Or ``{"error": ..., "available_skills": [...]}`` on miss.
        """
        skills = self._get_skills()
        if name not in skills:
            return {
                "error": f"Unknown skill '{name}'",
                "available_skills": list(skills.keys()),
            }

        info = skills[name]
        content = info["path"].read_text()
        metadata, body = _parse_frontmatter(content)
        intro, sections = _parse_skill_sections(body)
        refs = _list_skill_references(info["path"].parent)

        # Group H3s under their parent H2 so the TOC presents the
        # hierarchy. Slugs remain flat and unique within a skill —
        # callers look up sections by slug regardless of nesting.
        toc: list[dict] = []
        index: dict[str, dict] = {}
        for s in sections:
            entry = {"slug": s["slug"], "heading": s["heading"]}
            if s["level"] == 2:
                entry["subsections"] = []
                toc.append(entry)
                index[s["slug"]] = entry
            elif s["level"] == 3 and s["parent_slug"] in index:
                index[s["parent_slug"]]["subsections"].append(entry)
            else:
                toc.append(entry)
        # Drop empty subsections lists for cleaner output.
        for entry in toc:
            if entry.get("subsections") == []:
                entry.pop("subsections", None)

        return {
            "name": name,
            "description": metadata.get("description", ""),
            "intro": intro,
            "sections": toc,
            "references": refs,
            "next": (
                "Read one section: load_skill_section(name, slug) — works for any slug, H2 or H3. "
                "Read one reference file: load_skill_reference(name, ref). "
                "Don't pull material you don't need — slugs/refs are the menu."
            ),
        }

    def load_skill_section(self, name: str, slug: str) -> dict:
        """Load one section of a skill by its slug. Works for both H2
        (top-level) and H3 (nested under a How to use / Workflow H2).

        Args:
            name: Skill name (from ``list_skills`` / ``load_skill``).
            slug: Section slug as returned by ``load_skill(name).sections``
                (or as a subsection slug under any H2 entry).

        Returns:
            ``{"name", "section", "heading", "content"}`` on success;
            ``{"error", "available_sections": [...]}`` on miss.
        """
        skills = self._get_skills()
        if name not in skills:
            return {
                "error": f"Unknown skill '{name}'",
                "available_skills": list(skills.keys()),
            }
        content = skills[name]["path"].read_text()
        _, body = _parse_frontmatter(content)
        _, sections = _parse_skill_sections(body)
        for s in sections:
            if s["slug"] == slug:
                return {
                    "name": name,
                    "section": s["slug"],
                    "heading": s["heading"],
                    "content": s["content"],
                }
        return {
            "error": f"Unknown section '{slug}' in skill '{name}'",
            "available_sections": [s["slug"] for s in sections],
        }

    def load_skill_reference(self, name: str, ref: str) -> dict:
        """Load a reference file from ``<skill>/references/<ref>.md``.

        Args:
            name: Skill name.
            ref: Reference filename (with or without ``.md`` suffix).

        Returns:
            ``{"name", "reference", "content"}`` on success;
            ``{"error", "available_references": [...]}`` on miss.
        """
        skills = self._get_skills()
        if name not in skills:
            return {
                "error": f"Unknown skill '{name}'",
                "available_skills": list(skills.keys()),
            }
        skill_dir = skills[name]["path"].parent
        ref_name = ref[:-3] if ref.endswith(".md") else ref
        ref_path = skill_dir / "references" / f"{ref_name}.md"
        if not ref_path.is_file():
            return {
                "error": f"No reference '{ref_name}' in skill '{name}'",
                "available_references": _list_skill_references(skill_dir),
            }
        return {
            "name": name,
            "reference": ref_name,
            "content": ref_path.read_text(),
        }

    def create_skill(
        self,
        name: str,
        content: str,
        category: str = "",
    ) -> dict:
        """Create a new workflow skill in ~/.okonomi/skills/.

        The content must be a complete SKILL.md with YAML frontmatter containing
        at minimum 'name' and 'description' fields, followed by markdown instructions.
        Structure the body as ``## H2`` sections (the unit of progressive
        disclosure — load_skill lists them, load_skill_section fetches one) with
        distinct headings, and keep the preamble before the first H2 short.

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

        struct_err, warnings = _check_skill_structure(content)
        if struct_err:
            return {"success": False, "error": struct_err}

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
        result = {
            "success": True,
            "message": f"Skill '{name}' created.",
            "path": str(skill_dir),
        }
        if warnings:
            result["warnings"] = warnings
        return result

    def write_skill_file(self, name: str, file_path: str, content: str) -> dict:
        """Add or overwrite a file in a user skill — including SKILL.md.

        ``file_path`` of ``"SKILL.md"`` replaces the skill's main body
        and runs frontmatter validation. Any other path must live under
        ``references/``, ``templates/``, ``scripts/``, or ``assets/``.

        Args:
            name: Skill name.
            file_path: ``"SKILL.md"`` or a path under one of the
                allowed subdirs (e.g. ``"references/api.md"``).
            content: Full file contents.

        Returns:
            Dict with success status or error message.
        """
        skills = self._get_skills()
        if name not in skills:
            return {
                "success": False,
                "error": f"Skill '{name}' not found.",
                "available_skills": list(skills.keys()),
            }
        if skills[name].get("source") == "bundled":
            return {"success": False, "error": f"Cannot modify bundled skill '{name}'. Create a user copy first."}

        warnings: list[str] = []
        if file_path == "SKILL.md":
            err = _validate_frontmatter_content(content)
            if err:
                return {"success": False, "error": err}
            struct_err, warnings = _check_skill_structure(content)
            if struct_err:
                return {"success": False, "error": struct_err}
            target = skills[name]["path"]
        else:
            err = _validate_file_path(file_path)
            if err:
                return {"success": False, "error": err}
            target = skills[name]["path"].parent / file_path

        _atomic_write(target, content)
        self._invalidate_cache()
        result = {"success": True, "message": f"Wrote '{file_path}' in skill '{name}'."}
        if warnings:
            result["warnings"] = warnings
        return result

    def patch_skill_file(self, name: str, old_text: str, new_text: str, file_path: str = "SKILL.md") -> dict:
        """Surgical find-and-replace within a single file in a user
        skill. Cheap when you want to fix a typo or update one block;
        prefer this over ``write_skill_file`` to avoid resending a
        whole SKILL.md.

        ``old_text`` must occur exactly once. SKILL.md edits re-run
        frontmatter validation against the result.

        Args:
            name: Skill name.
            old_text: Text to find. Must be unique within the file.
            new_text: Replacement.
            file_path: File within the skill directory (default ``SKILL.md``).
        """
        skills = self._get_skills()
        if name not in skills:
            return {"success": False, "error": f"Skill '{name}' not found."}
        if skills[name].get("source") == "bundled":
            return {"success": False, "error": f"Cannot patch bundled skill '{name}'. Create a user copy first."}

        skill_dir = skills[name]["path"].parent
        target = skill_dir / file_path if file_path != "SKILL.md" else skills[name]["path"]
        if not target.exists():
            return {"success": False, "error": f"File '{file_path}' not found in skill '{name}'."}

        content = target.read_text()
        occurrences = content.count(old_text)
        if occurrences == 0:
            return {"success": False, "error": "old_text not found in file."}
        if occurrences > 1:
            return {
                "success": False,
                "error": f"old_text appears {occurrences} times — make it unique or use write_skill_file.",
            }

        new_content = content.replace(old_text, new_text, 1)
        warnings: list[str] = []
        if file_path == "SKILL.md":
            err = _validate_frontmatter_content(new_content)
            if err:
                return {"success": False, "error": err}
            struct_err, warnings = _check_skill_structure(new_content)
            if struct_err:
                return {"success": False, "error": struct_err}

        _atomic_write(target, new_content)
        self._invalidate_cache()
        result = {"success": True, "message": f"Patched '{file_path}' in skill '{name}'."}
        if warnings:
            result["warnings"] = warnings
        return result

    def delete_skill(self, name: str) -> dict:
        """Delete a user skill entirely (SKILL.md + every file under it).

        Args:
            name: Name of the skill to delete.
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
