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
<strong>Commit:</strong> {}
<strong>Circle build number:</strong> {}
<strong>Website build:</strong> {}
<strong>Website zip:</strong> {}
""".format(SHA1, JOB_ID, zip_url, web_url)

g = Github(GH_TOKEN)
repo = g.get_repo("XanaduAI/qml")
pr = repo.get_pull(PR)
pr.create_issue_comment(comment)
