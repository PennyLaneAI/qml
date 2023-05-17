# Demonstrations Metadata Guide

This document describes how to create the metadata file when writing a new demonstration.

## What is the metadata file any why do we have it?

Each demonstration has an associated metadata file. This file is a JSON file; it has the same name, but a different ending - always `.metadata.json`.

This file contains information that needs to be picked up by other systems and services. This is information like the title of the demonstration, the categories it's in, and the description and thumbnail image. By placing this information in a JSON file, it's easier for other systems - such as (and primarily) the pennylane.ai React website - to know what demonstrations exist, where they should be placed on the site, and how they are connected. You can think of the metadata file as 'registering' the demonstration with the system.

## How do you create the metadata file?

Generally, the simplest thing to do is to copy one of the existing metadata files and change the data in it. Some files may not use all of the fields that they can, so for a complete list of available fields, see [this page](https://github.com/PennyLaneAI/qml/blob/master/demonstrations_metadata.md).

Most of the fields should be self-explanatory, however, below is a list of things to remember while filling them in.

* The `title` field should exactly match the title given in the `.py` file.
* The `authors` field is a list of JSON objects, each of which has an `id`. This `id` field must exactly match the file name for the author in `_static/authors/`.
* The `dateOfPublication` and `dateOfLastModification` fields must be in the form `YYYY-MM-DDTHH:MM:SS`, but the hour, minute, and second values can just be set to `0`.
* The `categories` field is a list of strings. These strings will determine which categories the demonstration shows up in on the live site. The category string must exactly match that given for the title in `demonstrations/demonstrations_categories.metadata.json`.
* You can leave the `tags` field empty for now.
* The `previewImages` field is a list of images used in search results and other listings for the demonstration. One must be the default thumbnail to use, and must have the type `thumbnail`. Another can have the type `hero_image`, which is generally a larger version of the image. In all cases, the `uri` field must be a relative path to the image from the root of this repository.
* The `seoDescription` field should be the same as the description given in the Python file. This should ideally be close to 150 characters long (but not over), and end with a full stop.
* The `doi` field can be left empty for now.
* The `canonicalURL` field must be `/qml/demos/` + the file name of the demo without the `.py` ending.
* The `references` field is a list of references used by the demonstration. The fields within each reference match up to those used by BibTeX.
* If the demonstration is based on a very particular paper, put its DOI in the list on the `basedOnPapers` field.
* The `referencedByPapers` field can be left empty for now.
* The `relatedDemonstrations` field is a list of objects denoting which demonstrations this one is related to. In each of these objects, the only field that really needs to be different is `id`, which should be the file name of the related demonstration minus the file ending.

## A note about the file name of the demonstration.

The file name of the demonstration directly determines the URL at which the demonstration will be available. For SEO purposes, follow the rules below when naming the file.

* The file name should be the same as the title of the demonstration, but all lowercase, minus any punctuation marks, and replacing the spaces with underscores. (Don't cut out non-keywords such as 'of', 'the', 'and', and so on - use exactly the same words.)
* Don't use initialisms or abbreviations on their own unless they are extremely common in everyday language. Write the full term, and then put the initialism or abbreviation in brackets - i.e., 'A brief overview of the Variational Quantum Eigensolver (VQE)'.
