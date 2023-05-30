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
            "id": "guillermo_alonso"
        }
    ],
    "dateOfPublication": "2022-11-07T00:00:00",
    "dateOfLastModification": "2023-01-20T00:00:00",
    "categories": ["Getting Started"],
    "tags": ["quantum Fourier transforms", "qft"],
    "previewImages": [
        {
            "type": "thumbnail",
            "uri": "/_images/thumbnail_tutorial_qft_arithmetics.png"
        },
        {
            "type": "large_thumbnail",
            "uri": "/_static/large_demo_thumbnails/thumbnail_large_tutorial_<name>"
        },
        {
            "type": "hero_image",
            "uri": "/_static/hero_illustrations/qft_arithmetics_hero.png"
        }
    ],
    "seoDescription": "Learn how to use the quantum Fourier transform (QFT) to do basic arithmetic",
    "doi": "",
    "canonicalURL": "/qml/demos/tutorial_qft_arithmetics",
    "references": [
        {
            "id": "Draper2000",
            "type": "preprint",
            "title": "Addition on a Quantum Computer",
            "authors": "Thomas G. Draper",
            "year": "2000",
            "doi": "10.48550/arXiv.quant-ph/0008033",
            "url": "https://arxiv.org/abs/quant-ph/0008033"
        }
    ],
    "basedOnPapers": [],
    "referencedByPapers": [],
    "relatedDemonstrations": [
        {
            "type": "demonstration",
            "id": "tutorial_qubit_rotation",
            "weight": 1.0
        }
    ]
}
```



## Properties

The table below gives details about the fields in the metadata JSON file.

| Name | Is Required | Value Type | Description |
|---|---|---|---|
| `title` | Yes | `string` | The title of this demo. |
| `authors` | Yes | `array` of `object` | An array of the authors of this demo. This array must contain at least one item. See below for the object structure. |
| `dateOfPublication` | Yes | `datetime` | The date on which this demo was first published, in the form `YYYY-MM-DDTHH:MM:SS+00:00`. |
| `dateOfLastModification` | Yes | `datetime` | The date on which this demo was last modified, in the form `YYYY-MM-DDTHH:MM:SS+00:00`. |
| `categories` | Yes | `array` of `string` | An array of the categories that this demo is in. The currently-available categories are: `Getting Started`, `Optimization`, `QML`, `Quantum Chemistry`, `Quantum Computing`, and `Community`. |
| `tags` | Yes, but can be an empty array | `array` of `string` | An array of the tags that this demo has. |
| `previewImages` | Yes | `array` of `object` | An array of the different images that can be used as previews for this demo - e.g., thumbnails, social media cards (perhaps of different aspect ratios). See below for the object structure. |
| `seoDescription` | Yes | `string` | A description of the demo suitable for SEO purposes. Ideally this should be less than 150 characters, but this is not a strict limit. It should be a full, grammatically-correct sentence ending in a full stop. |
| `doi` | Yes, but can be an empty string | `string` | The DOI for this demo. |
| `canonicalURL` | Yes | `url` | The canonical URL for this demo. Sometimes there might be more than one URL that points to a given page on a website. The canonical URL defines which of these should be thought of as the _primary_ or _main_ one. |
| `references` | Yes | `array` of `object` | An array of the references used for this demo. See below for object structure. |
| `basedOnPapers` | Yes, but can be an empty array | `array` of `string` | An array of the DOIs for the papers this demo is based on. |
| `referencedByPapers` | Yes, but can be an empty array | `array` of `string` | An array of the DOIs of any papers that reference this demo. |
| `relatedContent` | Yes, but can be an empty array | `array` of `object` | An array of objects describing the content related to this demo. See below for the object structure. |

### Author Object Properties

| Name | Is Required | Value Type | Description |
|---|---|---|---|
| `id` | Yes | `string` | The id of this author. |
| `affiliation` | No | `string` | The affiliation of this author - often the university they work at. |

### Preview Image Object Properties

| Name | Is Required | Value Type | Description |
|---|---|---|---|
| `type` | Yes | `string` | What type of preview image this is. At the moment, the only value this can take is `thumbnail`, which refers to the image used on the QML part of pennylane.ai when browsing through lists of demos. |
| `uri` | Yes | `string` | The URI of this image, whether it be something hosted locally or on another site. |

### Reference Object Properties 

| Name | Is Required | Value Type | Description |
|---|---|---|---|
| `id` | Yes | `string` | An id for this reference used for citing it within the demo. |
| `type` | Yes | `string` | The type of this reference. Can be one of: `article`, `book`, `phdthesis`, `preprint`, `webpage` |
| `title` | Yes | `string` | The title of the paper or book being referenced. |
| `authors` | Yes | `string` | The authors of the paper or book being referenced, as a single string. Names should be comma-separated, and the Oxford comma should be used when there are 3 or more names, and before 'et al.'. |
| `year` | Yes | `string` | The year in which the paper or book was published. |
| `month` | No | `string` | The month in which the paper or book was published. |
| `journal` | No | `string` | The journal that the paper was published in. (Not relevant for books.) |
| `publisher` | No | `string` | The publisher of the book. (Not relevant for papers.) | 
| `doi` | No | `string` | The DOI of the paper. (Not the DOI URL - just the DOI.) |
| `url` | No | `string` | The URL of the paper or webpage. |
| `pages` | No | `string` | The specific pages of a journal or book being referenced, as a mixed list of individual pages and page ranges - i.e., `57, 61-63, 67, 102-104`. |
| `volume` | No | `string` | The volume of the journal or the multi-volume book. |
| `number` | No | `string` | The number of the journal. | 

### Related Content Object Properties

| Name | Is Required | Value Type | Description |
|---|---|---|---|
| `type` | Yes | `string` | The type of content that this relation refers to. So far, can only be `demonstration`, but this will be expanded in future. |
| `id` | Yes | `string` | The id of the content that this relation refers to. For demos, it's the file name of the demo without the extension - i.e., `tutorial_haar_measure`. |
| `weight` | Yes | `number` | A number between -1.0 and 1.0 indicating both how closely related these two pieces of content are, and which one it is preferable to encounter first. A value of 1.0 indicates that these two pieces of content are *very* closely related, and this one should be read first, and the linked one second. A value of -1.0 indicates again that these two pieces of content are very closely related, but that the linked one should be read first. A value of 0.0 indicates that these two pieces of content have nothing to do with each other. |




## An Empty Template

```json
{
    "title": "",
    "authors": [
        {
            "id": "",
            "affiliation": ""
        },
        {
            "id": "",
            "affiliation": ""
        }
    ],
    "dateOfPublication": "",
    "dateOfLastModification": "",
    "categories": [],
    "tags": [],
    "previewImages": [
        {
            "type": "thumbnail",
            "uri": ""
        }
    ],
    "seoDescription": "",
    "doi": "",
    "canonicalURL": "",
    "references": [
        {
            "id": "",
            "type": "",
            "title": "",
            "authors": "",
            "year": "",
            "journal": "",
            "doi": "",
            "url": ""
        },
        {
            "id": "",
            "type": "",
            "title": "",
            "authors": "",
            "year": "",
            "journal": "",
            "doi": "",
            "url": ""
        },
        {
            "id": "",
            "type": "",
            "title": "",
            "authors": "",
            "year": "",
            "journal": "",
            "doi": "",
            "url": ""
        }
    ],
    "basedOnPapers": [],
    "referencedByPapers": [
        {
            "type": "demonstration",
            "id": "",
            "weight": 1.0
        },
        {
            "type": "demonstration",
            "id": "",
            "weight": 1.0
        },
        {
            "type": "demonstration",
            "id": "",
            "weight": 1.0
        }
    ]
}
```

