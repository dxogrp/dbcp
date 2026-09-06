from __future__ import annotations

import json
import re
import runpy
import subprocess
import sys
from pathlib import Path

import pytest

import dbcp
from scripts.export_examples import export_examples
from scripts.stage_docs import documentation_series, stage_documentation_series

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DOCS_ROOT = REPOSITORY_ROOT / "docs"
EXAMPLES_ROOT = REPOSITORY_ROOT / "examples"
AUTODOC_PATTERN = re.compile(
    r"(?:```\{auto(?:class|function|exception|data|attribute)\}|\.\.\s+auto(?:class|function|exception|data|attribute)::)"
    r"\s+dbcp\.([A-Za-z_]\w*)",
    re.MULTILINE,
)
EXAMPLE_ROLE_PATTERN = re.compile(r"\{example\}`[^`]*<([^>]+)>`")


def _configuration() -> dict[str, object]:
    return runpy.run_path(str(DOCS_ROOT / "conf.py"))


def _make_build(directory: Path, marker: str, *, obsolete: bool = False) -> Path:
    directory.mkdir(parents=True)
    (directory / "index.html").write_text(marker, encoding="utf-8")
    assets = directory / "_static"
    assets.mkdir()
    (assets / "current.css").write_text(f"/* {marker} */", encoding="utf-8")
    if obsolete:
        (directory / "obsolete.html").write_text("obsolete", encoding="utf-8")
    return directory


def _switcher(site: Path) -> list[dict[str, object]]:
    return json.loads((site / "switcher.json").read_text(encoding="utf-8"))


def _snapshot(root: Path) -> tuple[tuple[str, str, bytes | None], ...]:
    entries: list[tuple[str, str, bytes | None]] = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if path.is_dir():
            entries.append(("directory", relative, None))
        else:
            entries.append(("file", relative, path.read_bytes()))
    return tuple(entries)


def _write_notebook(directory: Path, name: str = "example.py") -> Path:
    directory.mkdir(exist_ok=True)
    notebook = directory / name
    notebook.write_text(f"print({name!r})\n", encoding="utf-8")
    return notebook


def _write_fake_export(command: list[str]) -> subprocess.CompletedProcess[str]:
    Path(command[-1]).write_text(Path(command[-3]).stem, encoding="utf-8")
    return subprocess.CompletedProcess(command, 0, stdout="", stderr="")


def test_documentation_inventory_matches_public_api_and_examples() -> None:
    documentation = "\n".join(path.read_text(encoding="utf-8") for path in DOCS_ROOT.rglob("*.md"))
    documented = set(AUTODOC_PATTERN.findall(documentation))
    examples = sorted(path.stem for path in EXAMPLES_ROOT.glob("*.py"))
    linked_examples = sorted(EXAMPLE_ROLE_PATTERN.findall(documentation))

    assert documented == set(dbcp.__all__)
    assert examples
    assert linked_examples == examples


def test_documentation_configuration_uses_current_series_and_relative_links() -> None:
    configuration = _configuration()
    series = ".".join(dbcp.__version__.split(".")[:2])

    assert configuration["package_version"] == dbcp.__version__
    assert configuration["documentation_series"] == series
    assert configuration["version"] == configuration["release"] == series
    assert configuration["html_title"] == f"DBCP {series}"
    assert configuration["html_baseurl"] == f"https://dxogrp.github.io/dbcp/version/{series}/"
    assert configuration["extlinks"]["example"] == ("examples/%s.html", "%s")
    assert configuration["html_context"]["docs_switcher_url"] == "https://dxogrp.github.io/dbcp/switcher.json"


def test_export_examples_isolated_and_replaces_stale_tree(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _write_notebook(source, "b.py")
    _write_notebook(source, "a.py")
    (source / "zhlatex.mplstyle").write_text("axes.grid: True\n", encoding="utf-8")
    figures = source / "figures"
    figures.mkdir()
    (figures / "existing.pdf").write_bytes(b"existing")
    source_before = _snapshot(source)
    output = tmp_path / "rendered"
    output.mkdir()
    (output / "retired.html").write_text("stale", encoding="utf-8")
    calls: list[list[str]] = []

    def fake_runner(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        working_directory = Path(kwargs["cwd"])
        assert {path.name for path in working_directory.iterdir()} == {
            "a.py",
            "b.py",
            "zhlatex.mplstyle",
        }
        environment = kwargs["env"]
        assert isinstance(environment, dict)
        assert environment["MPLBACKEND"] == "Agg"
        assert environment["PYTHONOPTIMIZE"] == "0"
        return _write_fake_export(command)

    export_examples(source, output, runner=fake_runner)

    assert [Path(command[-3]).stem for command in calls] == ["a", "b"]
    assert calls[0][:5] == [sys.executable, "-m", "marimo", "export", "html"]
    assert {"--include-code", "--no-sandbox", "--force"} <= set(calls[0])
    assert sorted(path.name for path in output.iterdir()) == ["a.html", "b.html"]
    assert _snapshot(source) == source_before


def test_export_failure_preserves_previous_tree(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _write_notebook(source)
    output = tmp_path / "rendered"
    output.mkdir()
    current = output / "current.html"
    current.write_text("current", encoding="utf-8")

    def failing_runner(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        del kwargs
        raise subprocess.CalledProcessError(2, command, stderr="export error")

    with pytest.raises(RuntimeError, match=r"example\.py"):
        export_examples(source, output, runner=failing_runner)

    assert current.read_text(encoding="utf-8") == "current"
    assert not any(path.name.startswith(".rendered.") for path in tmp_path.iterdir())


def test_publish_helpers_reject_overlapping_trees(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _write_notebook(source)
    with pytest.raises(ValueError, match="must not overlap"):
        export_examples(source, source / "rendered")

    build = _make_build(tmp_path / "build", "unsafe")
    with pytest.raises(ValueError, match="must not overlap"):
        stage_documentation_series(build, build / "site", "0.2.3", "https://docs.example.test")


def test_documentation_series_requires_canonical_stable_version() -> None:
    assert documentation_series("0.2.3") == "0.2"
    assert documentation_series("10.20.30") == "10.20"

    invalid = ["0.2", "v0.2.3", "0.02.3", "0.2.3rc1", "0.2.3+local", "1!0.2.3"]
    for package_version in invalid:
        with pytest.raises(ValueError, match=r"canonical stable x\.y\.z"):
            documentation_series(package_version)


def test_first_series_populates_root_and_switcher(tmp_path: Path) -> None:
    build = _make_build(tmp_path / "build", "series 0.2")
    site = tmp_path / "site"
    git_head = site / ".git" / "HEAD"
    git_head.parent.mkdir(parents=True)
    git_head.write_text("ref: refs/heads/gh-pages", encoding="utf-8")

    stage_documentation_series(build, site, "0.2.3", "https://docs.example.test/dbcp/")

    assert (site / "version" / "0.2" / "index.html").read_text(encoding="utf-8") == "series 0.2"
    assert (site / "index.html").read_text(encoding="utf-8") == "series 0.2"
    assert (site / "_static" / "current.css").is_file()
    assert git_head.read_text(encoding="utf-8") == "ref: refs/heads/gh-pages"
    assert (site / ".nojekyll").is_file()
    assert _switcher(site) == [
        {
            "name": "latest",
            "version": "0.2",
            "url": "https://docs.example.test/dbcp/",
            "preferred": True,
        },
        {
            "name": "0.2",
            "version": "0.2",
            "url": "https://docs.example.test/dbcp/version/0.2/",
            "preferred": False,
        },
    ]


def test_documentation_series_lifecycle_replaces_and_promotes(tmp_path: Path) -> None:
    site = tmp_path / "site"
    initial = _make_build(tmp_path / "initial", "package 0.2.3", obsolete=True)
    replacement = _make_build(tmp_path / "replacement", "package 0.2.4")
    stage_documentation_series(initial, site, "0.2.3", "https://docs.example.test")
    stage_documentation_series(replacement, site, "0.2.4", "https://docs.example.test")

    current_series = site / "version" / "0.2"
    assert (current_series / "index.html").read_text(encoding="utf-8") == "package 0.2.4"
    assert (site / "index.html").read_text(encoding="utf-8") == "package 0.2.4"
    assert not (current_series / "obsolete.html").exists()
    assert not (site / "obsolete.html").exists()
    current_tree = _snapshot(current_series)

    older = _make_build(tmp_path / "older", "series 0.1")
    stage_documentation_series(older, site, "0.1.1", "https://docs.example.test")
    assert (site / "index.html").read_text(encoding="utf-8") == "package 0.2.4"
    assert _snapshot(current_series) == current_tree

    newer = _make_build(tmp_path / "newer", "series 0.3")
    stage_documentation_series(newer, site, "0.3.0", "https://docs.example.test")
    assert (site / "index.html").read_text(encoding="utf-8") == "series 0.3"
    assert _snapshot(current_series) == current_tree
    assert [entry["name"] for entry in _switcher(site)] == ["latest", "0.3", "0.2", "0.1"]
