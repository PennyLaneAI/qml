import subprocess
from collections import defaultdict
from pathlib import Path
import requirements
import tempfile
import itertools
from collections.abc import Mapping, Sequence


class RequirementsGenerator:
    """Generates 'requirements.txt' from 'requirements.in' files, with versions constrained
    by a global 'constraints.txt' file."""

    global_constraints: Mapping[str, Sequence[str]]
    extra_index_urls: Sequence[str] = ("https://download.pytorch.org/whl/cpu",)

    def __init__(
        self,
        python_bin: Path,
        global_constraints_file: Path,
        extra_index_urls: Sequence[str] | None = None,
        pre: bool = False,
    ):
        self.python_bin = python_bin

        global_constraints: dict[str, tuple[str, ...]] = defaultdict(tuple)
        with open(global_constraints_file, "r") as f:
            for req in requirements.parse(f):
                global_constraints[req.name] += (req.line,)

        self.global_constraints = global_constraints
        self._requirements_in_cache: dict[frozenset[str], str] = {}

    def generate_requirements(self, requirements_in: frozenset[str]):
        """Generate a 'requirements.txt' file for the given list of input requirements, with versions from
        global constraints. If any of the input requirements are version-qualified, the versions will
        override the global constraints.
        """
        if cached := self._requirements_in_cache.get(requirements_in):
            return cached

        constraints: dict[str, tuple[str, ...]] = defaultdict(tuple)
        for req_str in requirements_in:
            req = next(requirements.parse(req_str))
            if req.line != req.name:
                constraints[req.name] += (req.line,)

        for package in filter(
            lambda package: package not in constraints, self.global_constraints
        ):
            constraints[package] = self.global_constraints[package]

        with tempfile.TemporaryDirectory() as tmpdir:
            constraints_file = Path(tmpdir, "constraints.txt")
            requirements_file = Path(tmpdir, "requirements.txt")

            with open(constraints_file, "w") as f:
                for spec in itertools.chain.from_iterable(constraints.values()):
                    f.write(spec + "\n")

            with open(requirements_file, "w") as f:
                for req_str in requirements_in:
                    f.write(req_str + "\n")

            cmd = [
                str(self.python_bin),
                "-m",
                "uv",
                "pip",
                "compile",
                "--index-strategy",
                "unsafe-best-match",
                "--no-header",
                "--no-strip-extras",
                "--no-strip-markers",
                "--universal",
                "--quiet",
                "--no-annotate",
                "--emit-index-url",
                "--constraints",
                str(constraints_file),
                "--output-file",
                str(requirements_file),
            ]
            for index_url in self.extra_index_urls:
                cmd.extend(("--extra-index-url", index_url))

            cmd.append(str(requirements_file))

            subprocess.run(
                cmd,
            ).check_returncode()

            with open(requirements_file, "r") as f:
                reqs = f.read()
                self._requirements_in_cache[requirements_in] = reqs

                return reqs
