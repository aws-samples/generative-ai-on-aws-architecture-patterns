#!/bin/bash

# Set variables
aws_account_id=$(aws sts get-caller-identity --query Account --output text)
aws_region=$(aws configure get region)
repo_name=rag-app
image_name=latest
full_name="${aws_account_id}.dkr.ecr.${aws_region}.amazonaws.com/${repo_name}:${image_name}"

echo "AccountId = ${aws_account_id}"
echo "Region = ${aws_region}"

# If the repository doesn't exist in ECR, create it.
aws ecr describe-repositories --repository-names "${repo_name}" > /dev/null 2>&1

if [ $? -ne 0 ]
then
    echo "Creating ECR Repository..."
    aws ecr create-repository --repository-name "${repo_name}" > /dev/null
fi

# Get the login command for the new repository
echo "Logging into the repository..."
#$(aws ecr get-login --no-include-email)
aws ecr get-login-password --region ${aws_region} | docker login --username AWS --password-stdin ${full_name}

docker system prune -af

# Build and push the Docker image
echo "Building and pushing Docker image..."
docker build . -t ${image_name} -t ${full_name}

docker push ${full_name}