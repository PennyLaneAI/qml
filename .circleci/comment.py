import os
import requests

from github import Github

GH_TOKEN = os.environ["GH_AUTH_TOKEN"]
CIRCLE_TOKEN = os.environ["CIRCLE_TOKEN"]
JOB_ID = os.environ["CIRCLE_BUILD_NUM"]
SHA1 = os.environ["CIRCLE_SHA1"]
PR = int(os.environ["CIRCLE_PULL_REQUEST"].split("/")[-1])

HEADERS = {"Accept": "application/json", "Circle-Token": CIRCLE_TOKEN}
URL = "https://circleci.com/api/v2/project/github/XanaduAI/qml/{}/artifacts".format(JOB_ID)

res = requests.get(URL, headers=HEADERS).json()

zip_url = res["items"][0]["url"]

for f in res["items"]:
    if "html/index.html" in f["url"]:
        web_url = f["url"]
        break

comment = """\
<h3>Website build</h3>
<strong>Commit:</strong> {hash}\n
<strong>Circle build number:</strong> {job}\n
<strong>Website build:</strong> <a href={zip}>{zip}</a>\n
<strong>Website zip:</strong> <a href={web}>{web}</a>
""".format(hash=SHA1, job=JOB_ID, zip=zip_url, web=web_url)

g = Github(GH_TOKEN)
repo = g.get_repo("XanaduAI/qml")
pr = repo.get_pull(PR)

cmts = pr.get_issue_comments()
existing_comment = None

for c in cmts:
    if "<h3>Website build</h3>" in c.body and "josh146" == c.user.login:
        existing_comment = c
        break

if existing_comment is None:
    pr.create_issue_comment(comment)
else:
    existing_comment.edit(comment)
