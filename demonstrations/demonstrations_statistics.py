import json 
import glob 
import argparse


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

