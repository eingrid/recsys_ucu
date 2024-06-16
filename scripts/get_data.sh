#!/bin/bash

check_wget() {
    if ! command -v wget &> /dev/null; then
        echo "wget could not be found. Installing wget..."
        # Check for the OS and install wget accordingly
        if [ -x "$(command -v apt-get)" ]; then
            sudo apt-get update
            sudo apt-get install -y wget
        elif [ -x "$(command -v yum)" ]; then
            sudo yum install -y wget
        elif [ -x "$(command -v dnf)" ]; then
            sudo dnf install -y wget
        elif [ -x "$(command -v brew)" ]; then
            brew install wget
        else
            echo "Package manager not found. Please install wget manually."
            exit 1
        fi
    fi
}

download_dataset() {
    local url="https://files.grouplens.org/datasets/movielens/ml-1m.zip"
    local output="data/ml-1m.zip"

    wget "$url" -O "$output"
    if [ $? -ne 0 ]; then
        echo "Failed to download the dataset."
        exit 1
    fi
}

unzip_dataset() {
    local file="data/ml-1m.zip"
    local destination="data"

    unzip "$file" -d "$destination"
    if [ $? -ne 0 ]; then
        echo "Failed to unzip the dataset."
        exit 1
    fi
}

remove_zip() {
    local file="data/ml-1m.zip"

    rm "$file"
    if [ $? -ne 0 ]; then
        echo "Failed to remove the zip file."
        exit 1
    fi
}

# Create the data directory if it doesn't exist
mkdir -p data

check_wget
download_dataset
unzip_dataset
remove_zip

echo "Dataset downloaded and extracted successfully into the data folder."
