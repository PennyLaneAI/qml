import csv
import json 
import glob 
from datetime import datetime 


# The database downloaded from Notion. This has been excluded from the commits, and must be downloaded separately.
DEMONSTRATIONS_DATABASE = "demonstrations_database.csv"


def get_names():
    """ Gets a dictionary for which all of the keys are author names and all of the values are the author ids. """

    filePaths = glob.glob("_static/authors/*.txt")
    authors = {}

    # Go through all of the author file paths, open each file, and get the name out of it.
    for filePath in filePaths:
        with open(filePath, "r") as fileObject:
            lines = fileObject.readlines()

            authorId = filePath[16:-4]
            authorName = lines[0][8:].strip()

            authors[authorName] = authorId

    return authors 


def process_datetime(t):
    """ Just converts a date from the format 'd/m/y' to 'yyyy-mm-ddTHH:MM:SS'. """

    if t.strip() == "":
        return ""

    d = datetime.strptime(t, "%d/%m/%Y")

    return datetime.strftime(d, "%Y-%m-%dT00:00:00")


def update_metadata():
    """ 
    Updates all of the metadata files based on the contents of 
    the database from Notion, the contents of the author files, 
    and the contents of the demo Python files themselves. 
    """

    # Get the dictionary of author names and ids.
    names = get_names()

    n = 0

    with open(DEMONSTRATIONS_DATABASE, encoding="utf-8") as fo1:
        csvReader = csv.reader(fo1, delimiter=",")

        # Go through all of the rows in the database.
        for row in csvReader:

            # Some of the rows contain irrelevant data. If the first cell of a row starts with 'https://', then it's got demo data in it.
            if row[0].startswith("https://"):
                # Get the file name from the demo URL.
                fileName = row[0][31:-5]

                print(fileName)

                title = ""

                authors = row[5]
                # Get all of the author ids from the author dictionary.
                authors = [{"id": names.get(author.strip(), "")} for author in authors.split(",")]

                dateOfPublication = process_datetime(row[6])
                dateOfLastModification = process_datetime(row[7])
                categories = [row[9]]
                seoDescription = ""
                thumbnailURI = ""

                # Some data is easier to get from the demo file itself.
                with open("demonstrations/" + fileName + ".py", "r", encoding="utf-8") as fo2:
                    lines = fo2.readlines()

                    lastLine = ""

                    # By using some trickery with the contents of the lines, we can extract certain pieces of data.
                    for line in lines:
                        if line.strip().startswith("====="):
                            title = lastLine.strip()
                        if line.strip().startswith(":property=\"og:description\":"):
                            seoDescription = line.strip()[27:].strip()
                        if line.strip().startswith(":property=\"og:image\":"):
                            thumbnailURI = line.strip()[55:].strip()

                        lastLine = line 

                # Create a Python dictionary that we can export as JSON.

                demo = {}
                demo["title"] = title 
                demo["authors"] = authors 
                demo["dateOfPublication"] = dateOfPublication
                demo["dateOfLastModification"] = dateOfLastModification 
                demo["categories"] = categories 
                demo["tags"] = []
                demo["previewImages"] = [{"type": "thumbnail", "uri": thumbnailURI}]
                demo["seoDescription"] = ""
                demo["doi"] = ""
                demo["canonicalURL"] = row[0].strip()
                demo["references"] = []
                demo["basedOnPapers"] = []
                demo["referencedByPapers"] = []
                demo["relatedContent"] = []

                if n < 100:
                    metadataFileName = "demonstrations/" + fileName + ".metadata.json"
                    with open(metadataFileName, "w") as fo3:
                        json.dump(demo, fo3, indent=4)


                n += 1

    print(n)


def count_demos():
    """ Counts the number of demos based on the number of metadata files. """

    filePaths = glob.glob("demonstrations/*.metadata.json")

    print(len(filePaths))


if __name__ == "__main__":
    count_demos()