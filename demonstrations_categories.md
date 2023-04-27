# Demonstrations Categories

The file `demonstrations_categories.json` defines the categories we use for demonstrations. It gives the title of each category, the description (which is used on the `single category view page`), and the URL fragment.

### Category Definition Object Properties

| Name | Is Required | Value Type | Description |
|---|---|---|---|
| `title` | Yes | `string` | The title of this category. |
| `urlFragment` | Yes | `string` | The string to use in URLs that point to this category - i.e., if we had a URL like `pennylane.ai/qml/demonstrations/getting-started`, the `urlFragment` property defines that last part. |
| `description` | Yes | `string` | The description of this category. This is displayed on the `single category view page`. |

In order for a demonstration to be automatically shown in a category, the string in the `categories` property of the demonstrations metadata must exactly match the title of the category given in the definitions file.