# Dependency Specification

This directory houses the files that specify and constrain the dependencies required for QML demonstrations. These specifications ensure consistent and reproducible build and execution environments for all demos.

## Table of Contents

*   [Overview](#overview)
*   [Dependency Files](#dependency-files)
    *   [`constraints-dev.txt`](#constraints-devtxt)
    *   [`constraints-stable.txt`](#constraints-stabletxt)
    *   [`requirements-build.txt`](#requirements-buildtxt)
    *   [`requirements-core.in`](#requirements-corein)

---

## Overview

Managing dependencies is crucial for the reliability and reproducibility of our QML demonstrations. This directory contains dedicated files that define the precise versions of packages used in different contexts—ranging from development builds to stable releases and core execution requirements.

## Dependency Files

The following files define the various dependency sets:

### `constraints-dev.txt`

This file specifies the exact dependency versions used for building and running demonstrations when targeting the most recent **development builds** of PennyLane and its associated plugins. This ensures compatibility with cutting-edge features and ongoing development.

*   **Location:** `./constraints-dev.txt`
*   **Purpose:** Defines dependencies for the `qml build --dev` command.

### `constraints-stable.txt`

This file specifies the exact dependency versions used for building and running demonstrations against the most recent **stable release** of PennyLane and its associated plugins. This ensures compatibility and stability for production-ready demonstrations.

*   **Location:** `./constraints-stable.txt`
*   **Purpose:** Defines dependencies for the `qml build --no-dev` (default) command.

### `requirements-build.txt`

This file lists the dependencies specifically required for the build process of the demo notebooks using Sphinx. These packages are essential for converting and rendering the Python scripts into web-viewable formats (HTML/JSON).

*   **Location:** `./requirements-build.txt`
*   **Purpose:** Specifies tools needed for the Sphinx-based build system.

### `requirements-core.in`

This file specifies the fundamental execution dependencies that are automatically installed and available to **all** QML demonstrations by default. Demos can then specify additional, unique dependencies in their individual `requirements.in` files within their respective demo directories.

*   **Location:** `./requirements-core.in`
*   **Purpose:** Defines baseline dependencies for all demos.