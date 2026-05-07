from pathlib import Path
import subprocess


def test_legacy_model_artifact_is_not_referenced() -> None:
    legacy = "xgb" + "_pipe"
    root = Path(__file__).resolve().parents[1]
    tracked = subprocess.check_output(
        ["git", "ls-files"],
        cwd=root,
        text=True,
    ).splitlines()

    text_suffixes = {
        ".dockerignore",
        ".env",
        ".gitignore",
        ".ipynb",
        ".md",
        ".py",
        ".sh",
        ".toml",
        ".txt",
        ".yaml",
        ".yml",
    }
    offenders: list[str] = []

    for relative_path in tracked:
        path = root / relative_path
        if legacy in relative_path:
            offenders.append(relative_path)
            continue
        if path.suffix not in text_suffixes:
            continue
        content = path.read_text(encoding="utf-8", errors="ignore")
        if legacy in content:
            offenders.append(relative_path)

    assert offenders == []
