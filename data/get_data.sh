#!/bin/bash

# Function to check if wget is installed
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

# Function to download the dataset
download_dataset() {
    local url="https://files.grouplens.org/datasets/movielens/ml-1m.zip"
    local output="ml-1m.zip"

    wget "$url" -O "$output"
    if [ $? -ne 0 ]; then
        echo "Failed to download the dataset."
        exit 1
    fi
}

# Function to unzip the dataset
unzip_dataset() {
    local file="ml-1m.zip"

    unzip "$file" -d .
    if [ $? -ne 0 ]; then
        echo "Failed to unzip the dataset."
        exit 1
    fi
}

# Function to remove the zip file
remove_zip() {
    local file="ml-1m.zip"

    rm "$file"
    if [ $? -ne 0 ]; then
        echo "Failed to remove the zip file."
        exit 1
    fi
}

# Main script execution
check_wget
download_dataset
unzip_dataset
remove_zip

echo "Dataset downloaded and extracted successfully."
