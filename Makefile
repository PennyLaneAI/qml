# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line.
SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build/html
DATADIR       = _data

# Environment setup variables
POETRY_BIN     = poetry
POETRYOPTS     =
UPGRADE_PL     = false
BASE_ONLY      = false

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
	@echo
	@echo "Additional Targets:"
	@echo "  environment      Setup all packages and dependencies needed to build the docs"

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	$(SPHINXBUILD) -b html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
	@echo
	@echo "Build finished. The HTML pages are in $(BUILDDIR)."

html-norun:
	$(SPHINXBUILD) -D plot_gallery=0 -b html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
	@echo
	@echo "Build finished. The HTML pages are in $(BUILDDIR)."

json:
	$(SPHINXBUILD) -b json "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
	@echo
	@echo "Build finished. The JSON files are in $(BUILDDIR)."

json-norun:
	$(SPHINXBUILD) -D plot_gallery=0 -b json "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
	@echo
	@echo "Build finished. The JSON files are in $(BUILDDIR)."

text:
	$(SPHINXBUILD) -b text "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
	@echo
	@echo "Build finished. The text files are in $(BUILDDIR)."

text-norun:
	$(SPHINXBUILD) -D plot_gallery=0 -b text "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
	@echo
	@echo "Build finished. The text files are in $(BUILDDIR)."

download:
	# make data directories
	mkdir -p $(DATADIR)
	# download dataset for transfer learning tutorial
	wget --no-verbose -N https://download.pytorch.org/tutorial/hymenoptera_data.zip -P $(DATADIR)
	unzip -q -o $(DATADIR)/hymenoptera_data.zip -d $(DATADIR)/

environment:
	@command -v $(POETRY_BIN) --version >/dev/null 2>&1 || { echo >&2 "Setting up the environment requires a valid installation of python-poetry. Please install and add poetry to PATH or pass the executable using POETRY_BIN. Aborting."; exit 1; }
	@if [ '$(BASE_ONLY)' = 'true' ]; then\
		echo "Installing base Poetry dependencies ...";\
		$(POETRY_BIN) install --without executable-dependencies $(POETRYOPTS);\
	else\
		echo "Installing Poetry Dependencies ...";\
		$(POETRY_BIN) install $(POETRYOPTS);\
		if [ '$(UPGRADE_PL)' = 'true' ]; then\
			echo "Updating PennyLane and plugins to latest ... ";\
			PYTHON_VENV_PATH=`$(POETRY_BIN) env info --path`;\
			$$PYTHON_VENV_PATH/bin/python -m pip install --upgrade git+https://github.com/PennyLaneAI/pennylane.git@v0.38.0-rc0#egg=pennylane;\
			$$PYTHON_VENV_PATH/bin/python -m pip install --upgrade git+https://github.com/PennyLaneAI/pennylane-cirq.git#egg=pennylane-cirq;\
			$$PYTHON_VENV_PATH/bin/python -m pip install --upgrade git+https://github.com/PennyLaneAI/pennylane-qiskit.git#egg=pennylane-qiskit;\
			$$PYTHON_VENV_PATH/bin/python -m pip install --upgrade git+https://github.com/PennyLaneAI/pennylane-qulacs.git#egg=pennylane-qulacs;\
			$$PYTHON_VENV_PATH/bin/python -m pip install --extra-index-url https://test.pypi.org/simple/ PennyLane-Lightning --pre --upgrade;\
			$$PYTHON_VENV_PATH/bin/python -m pip install --extra-index-url https://test.pypi.org/simple/ PennyLane-Catalyst --pre --upgrade;\
		fi;\
	fi
