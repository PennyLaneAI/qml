import requests
from requests_auth_aws_sigv4 import AWSSigV4
import argparse
import sys
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DELETE_URL_TEMPLATE = "{endpoint_url}/demo/{slug}?pr_number={pr_number}"

parser = argparse.ArgumentParser(
    prog="delete",
    description="Delete deployed demos from the QML website.",
)

parser.add_argument(
    "slugs",
    help="The slug(s) of the demo(s) to delete.",
    type=str,
    nargs="+"
)
parser.add_argument(
    "-pr",
    "--pr-number",
    type=str,
    required=True,
    help="PR number for the preview version, or '0' for the main version (prod).",
)

def main():
    args = parser.parse_args()
    slugs = args.slugs
    pr_number = args.pr_number

    session = requests.Session()
    session.auth = AWSSigV4("execute-api", region="us-east-1")

    endpoint_url = os.environ["DEPLOYMENT_ENDPOINT_URL"]

    failed = False
    for slug in slugs:
        url = DELETE_URL_TEMPLATE.format(
            endpoint_url=endpoint_url, slug=slug, pr_number=pr_number
        )
        logger.info("Deleting demo '%s' (PR: %s) at '%s'", slug, pr_number, url)
        try:
            response = session.delete(url)
            response.raise_for_status()
            logger.info("Successfully deleted demo '%s' (PR: %s)", slug, pr_number)
        except requests.HTTPError:
            logger.error("Failed to delete demo '%s' (PR: %s) at '%s'", slug, pr_number, url, exc_info=True)
            failed = True

    if failed:
        sys.exit(1)

if __name__ == "__main__":
    main()


"""
Example usage:
    python delete.py demo-slug-1 demo-slug-2 -pr 42
"""