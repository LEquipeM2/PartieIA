#!/bin/bash

if [ -z "$1" ]; then
    echo "Usage: $0 <commit_message>"
    exit 1
fi

# Commandes Git
git add -v .
git reset -- amicorpus
git commit -m "$1"