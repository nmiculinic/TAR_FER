#!/usr/bin/env bash

echo "Downloading word embeddings..."
EMBEDDINGS_DIR="word_embeddings"
mkdir $EMBEDDINGS_DIR
wget https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.en.vec -P $EMBEDDINGS_DIR
echo "Done."
