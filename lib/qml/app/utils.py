import re


def slugify(text: str) -> str:
    """Convert a string to a slug-friendly format. This function handles CamelCase, removes special characters,
    replaces spaces and hyphens with underscores, and converts the string to lowercase.

    Examples:
    slugify("Hello World") -> "hello_world"
    slugify("CamelCaseExample") -> "camel_case_example"
    slugify("Special!@#Characters") -> "special_characters"
    slugify("Multiple   Spaces") -> "multiple_spaces"
    slugify("Hyphen-ated-Text") -> "hyphen_ated_text"
    """
    # Handle CamelCase by inserting spaces before uppercase letters
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    # Remove special characters except spaces and hyphens
    text = re.sub(r"[^\w\s-]", "", text)
    # Replace multiple spaces/hyphens with single underscore
    text = re.sub(r"[\s-]+", "_", text)
    # Convert to lowercase and strip
    return text.strip().lower()
