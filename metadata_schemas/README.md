# Demo Metadata

This folder contains the [metadata schemas](https://json-schema.org/) for the demos. The schema defines the metadata fields, their properties, and thier restrictions.

## Validating metadata

Install `poetry install --with metadata-validation`, then run the following from this directory:

```bash
  poetry run check-jsonschema --schemafile demo.metadata.schema.<version>.json ../demonstrations/*.metadata.json
```

## Generating the schema docs

Install `poetry install --with metadata-validation`, then run the following from this directory:

```bash
  poetry run generate-schema-doc demo.metadata.schema.<version>.json demo.metadata.schema.<version>.html
```

Open the resulting .html file to view the schema.
