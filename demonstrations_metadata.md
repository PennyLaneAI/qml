# Demonstrations Metadata

This document describes the structure of the JSON files that hold the metadata for the demonstrations.

The metadata JSON file for a given demo should be stored in the same folder as the Python demo file. It should have exactly the same name, except ending with `.metadata.json` instead of `.py`.

## Example

Below is given an example of a complete metadata file for a demonstration. 

```json
{
    "title": "Basic arithmetic with the quantum Fourier transform (QFT)",
    "authors": [
        {
            "name": "Guillermo Alonso-Linaje"
        }
    ],
    "dateOfPublication": "2022-11-07T00:00:00",
    "dateOfLastModification": "2023-01-20T00:00:00",
    "categories": ["Getting Started"],
    "tags": ["quantum Fourier transforms", "qft"],
    "previewImages": [
        {
            "type": "thumbnail",
            "uri": "/qml/_images/qft_arithmetics_thumbnail.png"
        }
    ],
    "seoDescription": "Learn how to use the quantum Fourier transform (QFT) to do basic arithmetic",
    "doi": "",
    "canonicalURL": "https://pennylane.ai/qml/demos/tutorial_qft_arithmetics.html",
    "references": [
        {
            "title": "Addition on a Quantum Computer",
            "authors": "Thomas G. Draper",
            "year": "2000",
            "journal": "",
            "doi": "arXiv:quant-ph/0008033"
        }
    ],
    "basedOnPapers": [],
    "referencedByPapers": []
}
```

## Properties

| Name | Is Required | Value Type | Description |
|---|---|---|---|
| `title` | Yes | `string` | The title of this demo. |
| `authors` | Yes | `array` of `object` | An array of the authors of this demo. This array must contain at least one item. See below for the object structure. |
| `dateOfPublication` | Yes | `datetime` | The date on which this demo was first published, in the form `YYYY-MM-DDTHH:MM:SS+00:00`. |
| `dateOfLastModification` | Yes | `datetime` | The date on which this demo was last modified, in the form `YYYY-MM-DDTHH:MM:SS+00:00`. |
| `categories` | Yes | `array` of `string` | An array of the categories that this demo is in. |
| `tags` | Yes, but can be an empty array | `array` of `string` | An array of the tags that this demo has. |
| `previewImages` | Yes | `array` of `object` | An array of the different images that can be used as previews for this demo - e.g., thumbnails, social media cards (perhaps of different aspect ratios). See below for the object structure. |
| `seoDescription` | Yes | `string` | A description of the demo suitable for SEO purposes. Ideally this should be less than 150 characters, but this is not a strict limit. |
| `doi` | Yes, but can be an empty string | `string` | The DOI for this demo. |
| `canonicalURL` | Yes | `url` | The canonical URL for this demo. Sometimes there might be more than one URL that points to a given page on a website. The canonical URL defines which of these should be thought of as the _primary_ one. |
| `references` | Yes | `array` of `object` | An array of the references used for this demo. See below for object structure. |
| `basedOnPapers` | Yes, but can be an empty array | `array` of `string` | An array of the DOIs for the papers this demo is based on. |
| `referencedByPapers` | Yes, but can be an empty array | `array` of `string` | An array of the DOIs of any papers that reference this demo. |

### Author Object Properties

| Name | Is Required | Value Type | Description |
|---|---|---|---|
| `name` | Yes | `string` | The name of this author. |
| `affiliation` | No | `string` | The affiliation of this author - often the university they work at. |

### Preview Image Object Properties

| Name | Is Required | Value Type | Description |
|---|---|---|---|
| `type` | Yes | `string` | What type of preview image this is. At the moment, the only value this can take is `thumbnail`, which refers to the image used on the QML part of pennylane.ai when browsing through lists of demos. |
| `uri` | Yes | `string` | The URI of this image, whether it be something hosted locally or on another site. |

### Reference Object Properties 

| Name | Is Required | Value Type | Description |
|---|---|---|---|
| `title` | Yes | `string` | The title of the paper being referenced. |
| `authors` | Yes | `string` | The authors of the paper being referenced, as a single string. |
| `year` | Yes | `string` | The year in which the paper was published. |
| `journal` | Yes | `string` | The journal that the paper was published in. |
| `doi` | Yes | `string` | The DOI of the paper. |


