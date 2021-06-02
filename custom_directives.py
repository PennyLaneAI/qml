# BSD 3-Clause License

# Copyright (c) 2017, Pytorch contributors
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.

# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.

# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
from docutils.parsers.rst import Directive, directives
from docutils.statemachine import StringList
from docutils import nodes
import re
import os
import sphinx_gallery

try:
    FileNotFoundError
except NameError:
    FileNotFoundError = IOError


GALLERY_TEMPLATE = """
.. raw:: html

    <div class="sphx-glr-thumbcontainer" data-category="{tags}" tooltip="{tooltip}">

.. only:: html

    .. figure:: {thumbnail}

        {description}

.. raw:: html

    </div>
"""


class CustomGalleryItemDirective(Directive):
    """Create a sphinx gallery style thumbnail.

    tooltip and figure are self explanatory. Description could be a link to
    a document like in below example.

    Example usage:

    .. customgalleryitem::
        :tooltip: I am writing this tutorial to focus specifically on NLP for people who have never written code in any deep learning framework
        :figure: /_static/img/thumbnails/babel.jpg
        :description: :doc:`/beginner/deep_learning_nlp_tutorial`

    If figure is specified, a thumbnail will be made out of it and stored in
    _static/thumbs. Therefore, consider _static/thumbs as a 'built' directory.
    """

    required_arguments = 0
    optional_arguments = 0
    final_argument_whitespace = True
    option_spec = {'tooltip': directives.unchanged,
                   'figure': directives.unchanged,
                   'description': directives.unchanged,
                   'tags': directives.unchanged}

    has_content = False
    add_index = False

    def run(self):
        try:
            if 'tooltip' in self.options:
                tooltip = self.options['tooltip'][:195]
            else:
                raise ValueError('tooltip not found')

            tags = ""
            if 'tags' in self.options:
                tags = self.options['tags']

            if 'figure' in self.options:
                env = self.state.document.settings.env
                rel_figname, figname = env.relfn2path(self.options['figure'])
                thumbnail = os.path.join('_static/thumbs/', os.path.basename(figname))

                try:
                    os.makedirs('_static/thumbs')
                except FileExistsError:
                    pass

                sphinx_gallery.gen_rst.scale_image(figname, thumbnail, 400, 280)
            else:
                thumbnail = '_static/thumbs/code.png'

            if 'description' in self.options:
                description = self.options['description']
            else:
                raise ValueError('description not doc found')

        except FileNotFoundError as e:
            print(e)
            return []
        except ValueError as e:
            print(e)
            raise
            return []

        thumbnail_rst = GALLERY_TEMPLATE.format(tooltip=tooltip,
                                                thumbnail=thumbnail,
                                                description=description,
                                                tags=tags)
        thumbnail = StringList(thumbnail_rst.split('\n'))
        thumb = nodes.paragraph()
        self.state.nested_parse(thumbnail, self.content_offset, thumb)
        return [thumb]


YOUTUBE_TEMPLATE = """
.. raw:: html

    <a href="https://youtube.com/watch?v={id}" target="_blank">
        <div class="card">
            <img class="card-img-top img-fluid" src="https://img.youtube.com/vi/{id}/hqdefault.jpg"/>
            <div class="card-body">
                <h4 class="card-title">{title}</h4>
                <p class="card-text grey-text">{author}</p>
                <p class="card-text">
                    {description}
                </p>
            </div>
            <div class="white-text watch">
                <hr>
                <h5>Watch <i class="fas fa-angle-double-right"></i></h5>
            </div>
        </div>
    </a>
"""


class YoutubeItemDirective(Directive):
    """Create a sphinx gallery style thumbnail.
    """

    required_arguments = 1
    optional_arguments = 2
    option_spec = {
        'title': directives.unchanged,
        'author': directives.unchanged}

    final_argument_whitespace = False
    has_content = True
    add_index = False

    def run(self):
        ytid = self.arguments[0]
        description = [i if i != "" else "<br><br>" for i in self.content]

        thumbnail_rst = YOUTUBE_TEMPLATE.format(id=ytid,
                                                title=self.options["title"],
                                                author=self.options["author"],
                                                description=" ".join(description))
        thumbnail = StringList(thumbnail_rst.split('\n'))
        thumb = nodes.paragraph()
        self.state.nested_parse(thumbnail, self.content_offset, thumb)
        return [thumb]


COMMUNITY_CARD_TEMPLATE = """
.. raw:: html

    <div class="card plugin-card" id={id}>
        <h4 class="card-header {color} lighten-4">{title}</h4>
        <div class="card-body">
            <h6>{author}</h6>
            <p class="font-small"><i class="far fa-clock pr-1"></i>{date}</p>
            <div class="row d-flex align-items-center">
                <div class="col-lg-8">
                    <p class="card-text">
                        {description}
                    </p>
                </div>
                <div class="col-lg-4">
                    {paper_footer}
                    {blog_footer}
                    {code_footer}
                </div>
            </div>
        </div>
    </div>
"""

PAPER_FOOTER = """<a href="{paper}" class="btn btn-info" style="box-shadow: unset; border-radius:5px; width:90%;">
                <i class="fas fa-book"></i> Paper
            </a>
"""

BLOG_FOOTER = """<a href="{blog}" class="btn btn-info" style="box-shadow: unset; border-radius:5px; width:90%;">
                <i class="fas fa-newspaper"></i> Blog
            </a>
"""

CODE_FOOTER = """<a href="{code}" class="btn btn-default" style="box-shadow: unset; border-radius:5px; width:90%;">
                <i class="fas fa-code-branch"></i></i> Code
            </a>
"""


class CommunityCardDirective(Directive):
    """Create a community card."""

    required_arguments = 0
    optional_arguments = 2
    option_spec = {
        'title': directives.unchanged,
        'author': directives.unchanged,
        'paper': directives.unchanged,
        'blog': directives.unchanged,
        'code': directives.unchanged,
        'date': directives.unchanged,
        'color': directives.unchanged,
    }

    final_argument_whitespace = False
    has_content = True
    add_index = False

    def run(self):
        description = [i if i != "" else "<br><br>" for i in self.content]
        color = self.options.get("color", "heavy-rain-gradient")
        code_footer = ""
        paper_footer = ""
        blog_footer = ""

        paper_url = self.options.get("paper", None)

        if paper_url is not None:
            paper_footer = PAPER_FOOTER.format(paper=paper_url)

        code_url = self.options.get("code", None)

        if code_url is not None:
            code_footer = CODE_FOOTER.format(code=code_url)

        blog_url = self.options.get("blog", None)

        if blog_url is not None:
            blog_footer = BLOG_FOOTER.format(blog=blog_url)

        def remove_accents(raw_text):
            """Removes common accent characters."""

            raw_text = re.sub(u"[àáâãäå]", 'a', raw_text)
            raw_text = re.sub(u"[èéêë]", 'e', raw_text)
            raw_text = re.sub(u"[ìíîï]", 'i', raw_text)
            raw_text = re.sub(u"[òóôõö]", 'o', raw_text)
            raw_text = re.sub(u"[ùúûü]", 'u', raw_text)
            raw_text = re.sub(u"[ýÿ]", 'y', raw_text)
            raw_text = re.sub(u"[ß]", 'ss', raw_text)
            raw_text = re.sub(u"[ñ]", 'n', raw_text)
            return raw_text

        card_rst = COMMUNITY_CARD_TEMPLATE.format(
            title=self.options["title"],
            author=self.options["author"],
            description=" ".join(description),
            date=self.options["date"],
            paper_footer=paper_footer,
            code_footer=code_footer,
            blog_footer=blog_footer,
            color=color,
            id=remove_accents(self.options["author"].split(" ")[-1].lower()) + self.options["date"].split("/")[-1] + self.options["title"].split(" ")[0].lower(),
        )

        thumbnail = StringList(card_rst.split('\n'))
        thumb = nodes.paragraph()
        self.state.nested_parse(thumbnail, self.content_offset, thumb)
        return [thumb]


RELATED = """
.. raw:: html

    <script type="text/javascript">
        var related_tutorials = [{urls}];
        var related_tutorials_titles = {linkText};
    </script>

"""


class RelatedDirective(Directive):
    """Add related demos to the sidebar.
    """

    required_arguments = 0
    optional_arguments = 0
    final_argument_whitespace = False
    has_content = True
    add_index = False

    def run(self):
        urls = [f"\"{u.split(' ')[0]}.html\"" for u in list(self.content)]
        linkText = [' '.join(u.split(' ')[1:]) for u in list(self.content)]
        urls = ", ".join(urls)
        html = RELATED.format(urls=urls, linkText=linkText)
        str_list = StringList(html.split('\n'))
        related_variables = nodes.paragraph()
        self.state.nested_parse(str_list, self.content_offset, related_variables)
        return [related_variables]
