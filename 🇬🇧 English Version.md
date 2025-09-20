# 🇬🇧 English Version

## Overview

This project is a minimal POC designed to showcase the setup of a recommendation system using the MovieLens 100k dataset

The goal is not to build a complex model but to demonstrate an end-to-end workflow:

Data extraction and preprocessing

Personalized recommendations generation

Exporting results

Deployment and execution on AWS EC2

👉 Output: for a given user, we obtain a top-k list of recommended movies with relevance scores.

## Tech stack

Python 3 (pandas, numpy)

MovieLens 100k (public dataset for movie recommendation)

AWS EC2 to run the project in the cloud

S3 for optional data storage

## Local usage

### Clone the repo:

git clone https://github.com/<https://github.com/Holyblitz/poc-aws-reco-ml100k>
cd poc-aws-reco-ml100k

pip install -r requirements.txt

### Install dependencies:

pip install -r requirements.txt

### Run a recommendation:

python poc_reco_kaggle_ml100k.py --user 42 --topk 10 --alpha 0.5

### Example output:

=== Recommendations for user 42 ===

Fargo (1996) — score 0.764

The Godfather (1972) — score 0.735

Pulp Fiction (1994) — score 0.727
...
CSV written: outputs/ml100k_recos_user_42.csv

## AWS Deployment

### Create SSH key

ssh-keygen -t rsa -b 4096 -f ~/.ssh/holyblitz-key

### Launch an Ubuntu instance

aws ec2 run-instances \
  --image-id <AMI_ID> \
  --count 1 \
  --instance-type t3.micro \
  --key-name <KEY_NAME> \
  --security-group-ids <SG_ID>

### Connect to the instance

ssh -i ~/.ssh/<KEY_NAME> ubuntu@<PUBLIC_IP>

## Deploy and run

Copy the project files, install dependencies, and run the script just like locally.
Results are exported in outputs/.

## Goal

Show the logic of a simple recommendation system.

Demonstrate Data Science + AWS Cloud integration.

Provide a foundation for more advanced projects (API, databases, scalability).



