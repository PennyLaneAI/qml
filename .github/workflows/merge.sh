#!/bin/bash

# This script is adapted from the robotology/gh-action-nightly-merge@v1.5.2 GitHub action:
# https://github.com/robotology/gh-action-nightly-merge/blob/master/entrypoint.sh

set -e

MERGE_HEAD=master
MERGE_BASE=dev
MERGE_ARGS="--no-ff --allow-unrelated-histories --no-edit"

git config --global --add safe.directory "$GITHUB_WORKSPACE"
git remote set-url origin https://x-access-token:$PUSH_TOKEN@github.com/$GITHUB_REPOSITORY.git
git config --global user.name "$CONFIG_USERNAME"
git config --global user.email "$CONFIG_EMAIL"

git fetch origin $MERGE_HEAD
(git checkout $MERGE_HEAD && git pull)||git checkout -b $MERGE_HEAD origin/$MERGE_HEAD

git fetch origin $MERGE_BASE
(git checkout $MERGE_BASE && git pull)||git checkout -b $MERGE_BASE origin/$MERGE_BASE

git merge $MERGE_ARGS $MERGE_HEAD
git push origin $MERGE_BASE