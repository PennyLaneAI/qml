import json 
import glob 
import argparse
import re 


DOI_PATTERN = r"\b(10[.][0-9]{4,}(?:[.][0-9]+)*/(?:(?![\"&\'<>])\S)+)\b"


def getAllMetadata():
    metadatas = {}
    filePaths = glob.glob("*.metadata.json")

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

    if arguments.action == "get_all_maths":
        fps = glob.glob("./demonstrations/*.py")

        print(len(fps))

        allMaths = {}

        for fp in fps:
            with open(fp, "r", encoding="utf-8") as fo:
                text = fo.read()

                ms = re.findall(":math:`([^`]+)`", text)

                for m in ms:
                    allMaths[m] = m

        allMathsList = sorted([k for k, v in allMaths.items()])

        with open("all_mathematical_expressions.txt", "w", encoding="utf-8") as fo:
            for line in allMathsList:
                fo.write(line + "\n")

        print(len(allMathsList))




