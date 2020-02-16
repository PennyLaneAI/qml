#!/usr/bin/env node

const bot = require("circle-github-bot").create();

bot.comment(`
<h3>${bot.env.commitMessage}</h3>
Website build: <strong>${bot.artifactLink('_build/html/index.html', 'Webpage')}</strong>
Website zip: <strong>${bot.artifactLink('/tmp/qml_html.zip', 'zip folder')}</strong>
`);
