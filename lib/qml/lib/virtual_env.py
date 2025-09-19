from pathlib import Path
import subprocess
import sys


class Virtualenv:
    """Interface to a Python virtual environment."""

    def __init__(self, path: Path):
        """
        Get python virtual env, creating it if it does not exist.
        Args:
            path: Path to virtual env directory. Will be initialized
                if it does not exist.
        """
        self.path = path.resolve()

        if not self.python.exists():
            self._init()

    @property
    def python(self) -> Path:
        """Path to the python executable in this virtual env."""
        return self.path / "bin" / "python"

    def _init(self):
        """Initialize a virtual environment."""
        self.path.parent.mkdir(exist_ok=True)

        subprocess.run([sys.executable, "-m", "venv", self.path]).check_returncode()
