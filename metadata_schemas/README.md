# Demo Metadata

**This directory contains the [metadata schema](https://json-schema.org/) for the demos.**

The schema defines the metadata fields as well as their properties and constraints.

## Validating Metadata

Install the necessary dependencies using:

```console
$ poetry install --only metadata-validation
```

Then run the following command **from this directory**:

```console
$ poetry run check-jsonschema --schemafile demo.metadata.schema.[VERSION].json ../demonstrations/*.metadata.json
```