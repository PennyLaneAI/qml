# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line.
SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build/html
DATADIR       = _data

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -b html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
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

download:
	# make data directories
	mkdir -p $(DATADIR)
	# download dataset for transfer learning tutorial
	wget --no-verbose -N https://download.pytorch.org/tutorial/hymenoptera_data.zip -P $(DATADIR)
	unzip -q -o $(DATADIR)/hymenoptera_data.zip -d $(DATADIR)/
