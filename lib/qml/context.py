from dulwich.repo import Repo
from pathlib import Path
import functools


class Context:
    """Context for CLI commands."""

    @property
    def repo_root(self) -> Path:
        """Absolute path to repository root."""
        return Path(self.repo.path).resolve()

    @property
    def demos_dir(self) -> Path:
        """Path to the content dir, relative to the current
        working directory."""
        return self.repo_root / "demonstrations_v2"

    @property
    def build_dir(self) -> Path:
        """Path to the build directory."""
        return self.repo_root / "_build"

    @property
    def build_venv_path(self) -> Path:
        """Path to virtual environment for building demos."""
        return self.repo_root / ".venv-build"

    @functools.cached_property
    def cwd(self) -> Path:
        """Current working directory of the process."""
        return Path.cwd().resolve()

    @functools.cached_property
    def repo(self) -> Repo:
        """dulwich ``Repo`` object for the repo."""
        return Repo.discover()

    @property
    def constraints_file(self) -> Path:
        return self.repo_root / "constraints.txt"
