import requests
from requests_auth_aws_sigv4 import AWSSigV4
import argparse
from pathlib import Path
import sys
import os
import logging

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)

DEPLOYMENT_URL_TEMPLATE = "{endpoint_url}/demo/{slug}?preview={preview}"

parser = argparse.ArgumentParser(
    prog="deploy",
    description="Deploy demo zip files to the QML website.",
)

parser.add_argument(
    "paths", help="The pattern to search for demo zip files.", type=Path, nargs="+"
)
parser.add_argument(
    "-p",
    "--preview",
    type=str,
    default="false",
    choices=("true", "false"),
    help="Whether to deploy to the preview site.",
)

def main():
    args = parser.parse_args()
    preview: str = args.preview
    paths: list[Path] = args.paths

    session = requests.Session()
    session.auth = AWSSigV4("execute-api", region="us-east-1")

    endpoint_url = os.environ["DEPLOYMENT_ENDPOINT_URL"]

    for path in paths:
        if not path.exists():
            logger.error("Path '%s' does not exist.", path)
            sys.exit(1)
        elif not (path.is_file() and path.suffix == ".zip"):
            logger.error("Path '%s' is not a zip file.", path)
            sys.exit(1)

        slug = path.stem
        url = DEPLOYMENT_URL_TEMPLATE.format(
            endpoint_url=endpoint_url, slug=slug, preview=preview
        )
        with open(path, "rb") as f:
            try:
                session.put(url, files={"file": f}).raise_for_status()
            except requests.HTTPError:
                logger.error("Failed to deploy '%s' to '%s'", path, url)
                sys.exit(1)

        logger.info("Deployed '%s' to '%s'", path, url)


if __name__ == "__main__":
    main()
