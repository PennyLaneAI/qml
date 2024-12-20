from pathlib import Path
import subprocess
import sys


class Virtualenv:
    def __init__(self, path: Path):
        self.path = path.resolve()

        if not self.python.exists():
            self.init()

    @property
    def python(self) -> Path:
        return self.path / "bin" / "python"

    def init(self):
        self.path.parent.mkdir(exist_ok=True)

        subprocess.run([sys.executable, "-m", "venv", self.path]).check_returncode()
