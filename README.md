# Generative AI retrieval-augmented generation (RAG) chatbot workshop

This **internal** repository contains instructions and examples to create your first generative AI chatbot connected to a knowledge base.

In this workshop you're going to learn:
- a foundational generative AI design pattern: retrieval augmented generation (RAG)
- how to use Amazon Kendra to implement a knowledge base with a document index
- how to use Amazon SageMaker JumpStart to deploy an open source LLM as a real-time endpoint and use the endpoint for inference
- how to use Amazon Bedrock to connect to large language models (LLMs) and use them via an API
- how to use popular Python frameworks [streamlit](https://streamlit.io/) and [langchain](https://python.langchain.com/docs/get_started/introduction) to implement generative AI applications in hourss

## Pre-requisites and environment setup
You need access to an AWS account. You can use your own isengard account or the shared isengard account, which is given by the workshop moderators.

If you use your own isengard account make sure to fulfill the following pre-requisites before the workshop:
1. Admin access to the account
2. Quota for `ml.g5.12xlarge` is set to at least 1. You can increase the quota in AWS console as described in [this instructions](https://aws.amazon.com/premiumsupport/knowledge-center/manage-service-limits/)
3. ?

### Clone the workshop repository
Clone the internal GitLab [repo](https://gitlab.aws.dev/ilyiny/genai-rag-bot-workshop) to your local notebook:

```sh
git clone git@ssh.gitlab.aws.dev:ilyiny/genai-rag-bot-workshop.git
```

[AWS Console access isengard](https://isengard.amazon.com/console-access)


### AWS Cloud9
- Setup AWS Cloud9 environment
- Upload the folder `genai-rag-bot-workshop` to the Cloud9 environment
- Open terminal in  Cloud9 and run `cd genai-rag-bot-workshop/`
- Run `aws sts get-caller-identity` and make sure you have the Administrator role
- Run `chmod +x resize-disk.sh`
- Run `./resize-disk.sh 100`





### SageMaker execution role permissions
You need the following permissions for your user profile execution role:
- `servicequotas:GetServiceQuota`
- `bedrock:*`




### Start SageMaker Studio
SageMaker domain
User profile -> for each individual user
Start SageMaker Studio


## Generative AI design patterns
Before you start the implementation of the RAG chatbot, let's go through some theory.

![](./static/img/rag-overview.png)


## Implementation

### Architecture overview
![]()

### Knowledge base

Create Kendra index
Ingest documents

### Retriever

### Generator

Use SageMaker JumpStart to create a LLM endpoint


### Chatbot app
The chatbot application source code is in the folder [`chatbot`](./chatbot/) of the workshop. This folder contain the following files: 
- `app.py`: the frontend utilizing the popular streamlit framework
- `Dockerfile`: Dockerfile providing the blueprint for the creation of a Docker image
- `requirements.txt`: specifying the dependencies required to be installed for hosting the frontend application
- `setup.sh`: setup script consisting all the necessary steps to create a ECR repository, build the Docker image and push it to the respective repository we created

In Cloud9 terminal:
- `cd chatbot/`
- `bash setup.sh`


## Resources
- [A Survey on In-context Learning](https://arxiv.org/abs/2301.00234) – a paper on in-context learning for LLMs
- [Deploy self-service question answering with the QnABot on AWS solution powered by Amazon Lex with Amazon Kendra and large language models](go.aws/48oG6WK) – in-depth example of RAG chatbot with Amazon Kendra
- [How to Build an Open-Domain Question Answering System?](bit.ly/3ZppYAl) – a good overview of information retrieval (IR) approaches
- [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401) – an original paper on RAG approach
- [Retrieval augmented generation (RAG)](https://docs.aws.amazon.com/sagemaker/latest/dg/jumpstart-foundation-models-customize-rag.html) – Amazon SageMaker Developer Guide
- [Implementing Generative AI on AWS workshop](https://catalog.us-east-1.prod.workshops.aws/workshops/80ae1ed2-f415-4d3d-9eb0-e9118c147bd4/en-US) – a public workshop


---

Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
SPDX-License-Identifier: MIT-0