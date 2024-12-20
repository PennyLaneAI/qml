from dulwich.repo import Repo
from pathlib import Path
import functools
from qml.lib.virtual_env import Virtualenv


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
        return self.repo_root / "_build"

    @property
    def build_venv_path(self) -> Path:
        return self.repo_root / ".venv-build"

    def build_venv(self) -> "Virtualenv":
        venv = Virtualenv(self.build_venv_path)

        return venv

    @functools.cached_property
    def cwd(self) -> Path:
        return Path.cwd().resolve()

    @functools.cached_property
    def repo(self) -> Repo:
        """dulwich ``Repo`` object for the repo."""
        return Repo.discover()
