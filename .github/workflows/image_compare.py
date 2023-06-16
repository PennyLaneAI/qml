import os
import subprocess

def main():
    """Parses two versions of the automatically run demonstrations from the QML
    repository, compares the output of each demo and writes to a file based on the
    differences found.
    """
    master_path = "/home/runner/work/qml/qml/demo_checker/master-build/_images/"
    dev_path = "/home/runner/work/qml/qml/demo_checker/dev-build/_images/"

    master_url = 'https://pennylane.ai/qml/_images/'
    dev_url = 'http://pennylane.ai-dev.s3-website-us-east-1.amazonaws.com/qml/_images/'

    master_rel_path = "./master-build/_images/"
    dev_rel_path = "./dev-build/_images/"

    # Get all the filenames
    master_files = os.listdir(master_path)
    # master_files.remove("thumb")
    dev_files = os.listdir(dev_path)

    #master_automatically_run = set([f for f in master_files if f.startswith("tutorial_")])
    #dev_automatically_run = set([f for f in dev_files if f.startswith("tutorial_")])

    #automatically_run = master_automatically_run.union(dev_automatically_run)

    output_file = open('demo_image_diffs.md','w')

    for f in master_files:
        if f[:4] != "sphx" or f[-9:] == "thumb.png":
            continue

        # Note: Imagemagick is required to be installed to perform
        # the following comparison
        command = ["identify", "-quiet", "-format", '"%#"']
        hash_master = subprocess.check_output(command + [master_path + f])
        hash_dev = subprocess.check_output(command + [dev_path + f])

        # Only consider images where the hashes differ
        if hash_master != hash_dev:
            # strip the Sphinx prefix ("sphx_glr_") and the image number and
            # extension suffix (e.g., "_001.png") to obtain the name of the demo
            demo_name = f[9:][:-8]
            image_number = f[-6:][:2]
            output_file.write(f'Demo: {demo_name}, image #: {image_number} \n\n')
            output_file.write(f'<img src="{master_rel_path + f}" alt="alt text" title="image Title" height="300"/> \n\n')

            output_file.write(f'<img src="{dev_rel_path + f}" alt="alt text" title="image Title" height="300"/>')
            output_file.write(f'\n\n---\n\n')

    return 0

if __name__ == '__main__':
    main()
