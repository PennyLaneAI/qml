from dataclasses import dataclass
from pathlib import Path
from collections.abc import Sequence, Iterator


@dataclass
class Demo:
    name: str
    path: Path

    @property
    def py_file(self) -> Path:
        return self.path / "demo.py"

    @property
    def metadata_file(self) -> Path:
        return self.path / "metadata.json"

    @property
    def requirements_file(self) -> Path:
        return self.path / "requirements.in"

    @property
    def resources(self) -> Sequence[Path]:
        return tuple(
            p
            for p in self.path.iterdir()
            if p not in {self.py_file, self.metadata_file, self.requirements_file}
        )

    @property
    def executable(self) -> bool:
        return self.name.startswith("tutorial_")

    def requirements(self):
        with open(self.requirements_file, "r") as f:
            return f.read().splitlines()


def find_demos(search_dir: Path, *names: str) -> Iterator[Demo]:
    if not names:
        yield from (
            Demo(name=demo_dir.name, path=demo_dir.resolve())
            for demo_dir in search_dir.iterdir()
            if demo_dir.is_dir()
        )

        return

    for name in set(names):
        demo_dir = search_dir / name
        if not (demo_dir / "demo.py").exists():
            raise ValueError(f"No demo exists with name '{name}")

        yield Demo(name=name, path=demo_dir.resolve())
