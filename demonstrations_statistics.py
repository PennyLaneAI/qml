import json 
import glob 
import argparse
import re 
import datetime 


DOI_PATTERN = r"\b(10[.][0-9]{4,}(?:[.][0-9]+)*/(?:(?![\"&\'<>])\S)+)\b"


def getAllMetadata():
    metadatas = {}
    filePaths = glob.glob("demonstrations/*.metadata.json")

    for filePath in filePaths:
        i2 = filePath.find(".metadata")
        fileName = filePath[:i2]

        with open(filePath, "r", encoding="utf-8") as fo:
            metadata = json.load(fo)

            metadatas[fileName] = metadata 

    return metadatas 


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--action")
    parser.add_argument("--title-1")
    parser.add_argument("--title-2")

    arguments = parser.parse_args()

    if arguments.action == "count":
        metadatas = getAllMetadata()
        print(len(metadatas))
    
    if arguments.action == "count_per_year":
        metadatas = getAllMetadata()
        perYear = []

        for year in [2018, 2019, 2020, 2021, 2022, 2023]:
            perYear.append({"Year": year, "Count": len([d for k, d in metadatas.items() if d["dateOfPublication"].startswith(str(year))])})

        for year in perYear:
            print("{0}: {1}".format(year["Year"], year["Count"]))

    if arguments.action == "check":
        metadatas = getAllMetadata()

        for name, metadata in metadatas.items():
            if not metadata["seoDescription"].endswith("."):
                pass 
                #print(name)
            if len(metadata["categories"]) == 0:
                pass 
                #print("{0} is not in any category.".format(name))


            for doi in metadata["basedOnPapers"]:
                if doi != "" and not re.match(DOI_PATTERN, doi):
                    print("{0} has an incorrectly-formatted DOI.".format(name))

            for reference in metadata["references"]:
                doi = reference.get("doi", "")
                
                if doi != "" and not re.match(DOI_PATTERN, doi):
                    print("{0} has an incorrectly-formatted DOI.".format(name))

    if arguments.action == "retitle-category":
        title1 = arguments.title_1.strip()
        title2 = arguments.title_2.strip()

        fps = glob.glob("./demonstrations/*.metadata.json")

        for fp in fps:
            with open(fp, "r", encoding="utf-8") as fo:
                metadata = json.load(fo)

            metadata["categories"] = [title2 if c.strip() == title1 else c.strip() for c in metadata["categories"]]

            with open(fp, "w", encoding="utf-8") as fo:
                json.dump(metadata, fo, indent=4, ensure_ascii=False)

    if arguments.action == "get_all_categories_used":

        fps = glob.glob("./demonstrations/*.metadata.json")
        categories = {}

        for fp in fps:
            with open(fp, "r", encoding="utf-8") as fo:
                metadata = json.load(fo)

                for category in metadata["categories"]:
                    if category.strip() != "":
                        categories[category] = category 

        print([k for k, v in categories.items()])

    if arguments.action == "get_most_recent_demos":
        metadata = getAllMetadata()
        mostRecent = [v for k, v in metadata.items()]
        mostRecent = sorted(mostRecent, key=lambda m: datetime.datetime.strptime(m["dateOfPublication"], "%Y-%m-%dT%H:%M:%S"), reverse=True)

        for m in mostRecent[:5]:
            #print(m)
            print(m["title"] + ", " + m["dateOfPublication"])








