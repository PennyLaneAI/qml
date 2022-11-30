#!/usr/bin/env python3

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

setup(
    name="qml_pipeline_utils",
    version="1",
    url="https://github.com/PennyLaneAI/qml",
    packages=["qml_pipeline_utils", "qml_pipeline_utils.services"],
    entry_points={"console_scripts": ["qml_pipeline_utils=qml_pipeline_utils.cli:cli_parser"]},
)
