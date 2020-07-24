import textwrap
import sys

if __name__ == "__main__":
    commit_hash, url = sys.argv[1:]

    comment = textwrap.dedent(f"""\
    <h3>Website build</h3>
    <strong>Commit:</strong> {commit_hash}\n
    <strong>Website url:</strong> <a href={url}>{url}</a>\n
    <em>Please double check the rendered website build to make sure everything is correct.</em>
    """)

    print(comment)
