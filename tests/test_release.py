from __future__ import annotations

import hashlib
import importlib.metadata
import io
import tarfile
import tomllib
import zipfile
from pathlib import Path

import pytest

from scripts.verify_release import canonical_stable_version, verify_distributions, write_checksums

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _source_tree(root: Path) -> Path:
    source = root / "src" / "dbcp"
    source.mkdir(parents=True)
    (source / "__init__.py").write_text(
        'from importlib.metadata import version\n\n__version__ = version("dbcp")\n',
        encoding="utf-8",
    )
    (source / "problem.py").write_text("class BiconvexProblem: ...\n", encoding="utf-8")
    return source


def _wheel(
    dist: Path,
    *,
    version: str = "0.1.0",
    metadata_version: str | None = None,
    tag: str = "py3-none-any",
) -> Path:
    path = dist / f"dbcp-{version}-{tag}.whl"
    metadata = metadata_version or version
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(
            "dbcp/__init__.py",
            'from importlib.metadata import version\n\n__version__ = version("dbcp")\n',
        )
        archive.writestr("dbcp/problem.py", "class BiconvexProblem: ...\n")
        archive.writestr(
            f"dbcp-{version}.dist-info/METADATA",
            f"Metadata-Version: 2.4\nName: dbcp\nVersion: {metadata}\nRequires-Python: >=3.12\n\n",
        )
        archive.writestr(
            f"dbcp-{version}.dist-info/WHEEL",
            f"Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: {tag}\n\n",
        )
        archive.writestr(f"dbcp-{version}.dist-info/licenses/LICENSE", "Apache License\n")
    return path


def _add_tar_bytes(archive: tarfile.TarFile, name: str, data: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(data)
    archive.addfile(info, io.BytesIO(data))


def _sdist(dist: Path, source: Path, *, version: str = "0.1.0", extra: str | None = None) -> Path:
    path = dist / f"dbcp-{version}.tar.gz"
    root = f"dbcp-{version}"
    pyproject = f'[project]\nname = "dbcp"\nversion = "{version}"\n'
    with tarfile.open(path, "w:gz") as archive:
        _add_tar_bytes(archive, f"{root}/LICENSE", b"Apache License\n")
        _add_tar_bytes(
            archive,
            f"{root}/PKG-INFO",
            f"Metadata-Version: 2.4\nName: dbcp\nVersion: {version}\nRequires-Python: >=3.12\n\n".encode(),
        )
        _add_tar_bytes(archive, f"{root}/README.md", b"# DBCP\n")
        _add_tar_bytes(archive, f"{root}/pyproject.toml", pyproject.encode())
        for path_source in source.glob("*.py"):
            _add_tar_bytes(archive, f"{root}/src/dbcp/{path_source.name}", path_source.read_bytes())
        if extra is not None:
            _add_tar_bytes(archive, f"{root}/{extra}", b"unexpected\n")
    return path


@pytest.mark.parametrize("version", ["0.1.0", "1.2.3", "10.20.30"])
def test_canonical_stable_release_versions(version: str) -> None:
    assert str(canonical_stable_version(version)) == version


@pytest.mark.parametrize(
    "version",
    ["0.0.0", "v0.1.0", "0.1", "01.2.3", "1!0.2.0", "0.2.0rc1", "0.2.0.post1", "0.2.0.dev1", "0.2.0+local"],
)
def test_noncanonical_unstable_or_unreleased_versions_are_rejected(version: str) -> None:
    with pytest.raises(ValueError, match="release version|Release version"):
        canonical_stable_version(version)


def test_release_distributions_and_checksums(tmp_path: Path) -> None:
    source = _source_tree(tmp_path)
    dist = tmp_path / "dist"
    dist.mkdir()
    wheel = _wheel(dist)
    sdist = _sdist(dist, source)

    assert verify_distributions(dist, "0.1.0", source) == (wheel, sdist)

    checksums = dist / "SHA256SUMS"
    write_checksums((wheel, sdist), checksums)
    expected = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in (wheel, sdist)}
    actual = {
        name: digest
        for digest, name in (line.split("  ", 1) for line in checksums.read_text(encoding="utf-8").splitlines())
    }
    assert actual == expected


def test_release_verifier_rejects_inconsistent_wheel_metadata(tmp_path: Path) -> None:
    source = _source_tree(tmp_path)
    dist = tmp_path / "dist"
    dist.mkdir()
    _wheel(dist, metadata_version="0.2.0")
    _sdist(dist, source)

    with pytest.raises(ValueError, match="wrong project version"):
        verify_distributions(dist, "0.1.0", source)


def test_release_verifier_rejects_nonuniversal_wheel(tmp_path: Path) -> None:
    source = _source_tree(tmp_path)
    dist = tmp_path / "dist"
    dist.mkdir()
    _wheel(dist, tag="cp312-cp312-manylinux_2_17_x86_64")
    _sdist(dist, source)

    with pytest.raises(ValueError, match="py3-none-any"):
        verify_distributions(dist, "0.1.0", source)


@pytest.mark.parametrize("extra", ["tests/test_release.py", "src/dbcp/__pycache__/problem.pyc"])
def test_release_verifier_rejects_development_or_generated_files_in_sdist(tmp_path: Path, extra: str) -> None:
    source = _source_tree(tmp_path)
    dist = tmp_path / "dist"
    dist.mkdir()
    _wheel(dist)
    _sdist(dist, source, extra=extra)

    with pytest.raises(ValueError, match="development-only path|Generated file"):
        verify_distributions(dist, "0.1.0", source)


def test_release_verifier_rejects_unsafe_sdist_paths(tmp_path: Path) -> None:
    source = _source_tree(tmp_path)
    dist = tmp_path / "dist"
    dist.mkdir()
    _wheel(dist)
    _sdist(dist, source, extra="../escape.py")

    with pytest.raises(ValueError, match="Unsafe path"):
        verify_distributions(dist, "0.1.0", source)


def test_hatch_sdist_manifest_is_minimal() -> None:
    project = tomllib.loads((REPOSITORY_ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    assert project["tool"]["hatch"]["build"]["targets"]["sdist"] == {
        "only-include": ["src/dbcp", "README.md", "LICENSE", "pyproject.toml"]
    }


def test_runtime_version_comes_from_distribution_metadata() -> None:
    import dbcp

    assert dbcp.__version__ == importlib.metadata.version("dbcp")
