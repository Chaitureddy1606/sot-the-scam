#!/bin/bash

# Create data directory if it doesn't exist
mkdir -p data

# Download the dataset
echo "Downloading the EMSCAD dataset..."
curl -L "https://storage.googleapis.com/kaggle-datasets-public/fake_job_postings.csv" -o data/fake_job_postings.csv

# Alternative URLs if the above fails
if [ ! -s data/fake_job_postings.csv ]; then
    echo "Trying alternative source..."
    curl -L "https://archive.ics.uci.edu/ml/machine-learning-databases/00343/fake_job_postings.csv" -o data/fake_job_postings.csv
fi

# Verify the download
if [ -f "data/fake_job_postings.csv" ] && [ -s data/fake_job_postings.csv ]; then
    echo "Dataset downloaded successfully!"
    echo "File size: $(ls -lh data/fake_job_postings.csv | awk '{print $5}')"
    echo "Number of rows: $(wc -l < data/fake_job_postings.csv)"
    
    # Display first few lines to verify content
    echo -e "\nFirst few lines of the dataset:"
    head -n 3 data/fake_job_postings.csv
else
    echo "Error: Failed to download the dataset"
    echo "Please download the dataset manually from Kaggle:"
    echo "https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction"
    echo "and place it in the data/ directory as fake_job_postings.csv"
    exit 1
fi 