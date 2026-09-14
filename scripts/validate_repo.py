from __future__ import annotations

from pathlib import Path

from wre.repo_validation import validate_repository

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    errors = validate_repository(ROOT)
    if not errors:
        print("repository metadata: OK")
        return 0

    print("repository metadata validation failed:")
    for error in errors:
        print(f"- {error}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
