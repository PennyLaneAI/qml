"""
Custom Sphinx extension that adds text translation support for the ``image-sg``
directive from Sphinx Gallery.
"""

from sphinx.errors import ExtensionError
from sphinx.util.docutils import is_node_registered
from sphinx_gallery.directives import imgsgnode

def visit_imgsg_text(self, node):
    """Visit function for ``imgsgnode`` nodes."""
    self.visit_image(node)

def depart_imgsg_text(self, node):
    """Departure function for ``imgsgnode`` nodes."""
    self.depart_image(node)

def setup(app):
    """Entrypoint for the extension."""
    if not is_node_registered(imgsgnode):
        raise ExtensionError("Sphinx Gallery extension must be loaded first.")

    app.registry.add_translation_handlers(
        imgsgnode,
        text=(visit_imgsg_text, depart_imgsg_text),
    )
