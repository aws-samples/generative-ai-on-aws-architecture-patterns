# Generative AI on AWS architecture patterns

In this workshop you're going to learn:
- foundational design patterns to implement a broad range of generative AI applications - from simple chatbots to complex multi-agent architectures integrated with external tools and systems
- how to use Amazon Bedrock to connect to foundation models and use them via an API
- how to use Amazon Kendra to implement a knowledge base with a document index
- how to ingest documents into the knowledge base using Kendra connectors
- how to use Amazon SageMaker JumpStart to deploy an open source large language model (LLM) as a real-time endpoint and use the endpoint for text generation  
- how to use HuggingFace TGI container and Amazon SageMaker to deploy an LLM as a real-time endpoint
- how to use popular Python frameworks [streamlit](https://streamlit.io/) and [LangChain](https://python.langchain.com/docs/get_started/introduction) to implement generative AI applications 
- how to leverage AWS services to abstract complex functionality and build your generative AI applications fast and without undifferentiated heavy lifting 

## Who this workshop is for
This workshop is for developers, designers, and architects who build enterprise applications and interested in improving their understanding of generative AI and want to develop applications leveraging LLMs. We created the workshop with a focus to present the most common emerging and established generative AI architectural patterns corresponding to a wide range of possible business use cases.

### Recommended knowledge and experience
Understanding and practical experience with the following AWS services are recommended:
- Amazon Elastic Container Service
- Amazon SageMaker
- Amazon Kendra
- AWS Lambda
- Amazon OpenSearch
- Amazon API Gateway
- Amazon CloudFormation

We also strongly recommend hands-on experience with general application development on AWS, also using AWS Serverless Application Model and AWS CloudFormation templates.

Previous experience with generative AI, Large Language Models (LLMs) and Foundational Models (FMs) is not required, you can learn as you work through the workshop.

## How to use this workshop
The workshop consists of simple self-contained labs covering the foundational architecture patterns for generative AI application and providing implementation of those patterns on AWS.

## Pre-requisites and environment setup
You need access to an AWS account. You can use your own account or the shared account, which is given by the workshop moderators.

If you use your own account make sure to fulfill the following pre-requisites before the workshop:
1. Admin access to the account
1. [Request access](https://docs.aws.amazon.com/bedrock/latest/userguide/model-access.html) to Amazon Bedrock models
1. [AWS Console access](https://console.aws.amazon.com/console/home?#)
1. If you'd like to experiment with an LLM real-time SageMaker endpoints, you'll need the following SageMaker ML instances for real-time inference:
- minimal setup to be able to deploy and experiment with small models, like [Falcon-7B](https://huggingface.co/tiiuae/falcon-7b-instruct): `ml.g5.2xlarge`
- recommended setup to be able to use bigger models like [Falcon-40B](https://huggingface.co/tiiuae/falcon-40b-instruct): `ml.g5.12xlarge` (4x NVidia GPUs) or `ml.g5.48xlarge` (8x NVidia GPUs)

You can check and increase your quotas for these instances as described in [this instructions](https://aws.amazon.com/premiumsupport/knowledge-center/manage-service-limits/). You don't need to increase quota if you're going to use Amazon Bedrock only

![](/static/img/sagemaker-quota-increase.png)

The next sections contain step-by-step instructions how to setup the required development environments.

### Setup AWS Cloud9 environment
You're going to use AWS Cloud9 environment to prepare the chatbot app container and build and deploy the end-to-end application stack.

To setup AWS Cloud9:
- Sign in an AWS account 
- If you don't have an existing AWS Cloud9 environment, setup a new one. See [Hello AWS Cloud9](https://docs.aws.amazon.com/cloud9/latest/user-guide/tutorial.html) for a step-by-step instruction
- Choose at least `m5.large` instance and `Ubuntu Server 22.04 LTS` as the platform
- Open the AWS Cloud9 environment

You're going to use Python 3.11 to build generative AI application. Run the following commands in AWS Cloud9 terminal.

Upgrade Python version to 3.11:  

1. List the installed python versions  
```sh
ls -l /usr/bin/python*
```

2. Install Python 3.11  
```sh
sudo apt update -y
sudo apt install python3.11 -y
```

3. See the default Python version  
```sh
python3 --version
```

4. Set 3.11 as the default Python  
```sh
sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1
sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 2
sudo update-alternatives --config python3
```

5. Check that the default Python version is 3.11  
```sh
python3 --version
```

Upgrade AWS SAM CLI:
```sh
pip install --upgrade aws-sam-cli
```

Check the AWS SAM CLI version:
```sh
sam --version
```

#### Download source code into your development environment
To have all workshop assets you must clone the workshop source code repository into your AWS Cloud9 environment.

In the AWS Cloud9 terminal window:

- Clone the workshop github repo to your environment:
`git clone https://github.com/aws-samples/generative-ai-on-aws-architecture-patterns.git`
- Change to the workshop folder: 
`cd ~/environment/generative-ai-on-aws-architecture-patterns/`
- Increase the size of AWS Cloud9 EBS volume:  
`bash ./content/scripts/resize-disk.sh 100`

### Setup SageMaker Studio
If you're going to use SageMaker Studio to deploy a real-time LLM endpoint, you need to provision a SageMaker domain and setup SageMaker Studio.

❗ You don't need Studio and the endpoint if you use Amazon Bedrock API to connect to an LLM. 
You can also skip all SageMaker-related sections if you're going to use Amazon Bedrock only.

To setup Studio:
1. [Provision a SageMaker domain](https://docs.aws.amazon.com/sagemaker/latest/dg/gs-studio-onboard.html) if you don't have one
2. [Provision a user profile](https://docs.aws.amazon.com/sagemaker/latest/dg/domain-user-profile-add-remove.html) if you don't have one

#### SageMaker execution role permissions
You need the following permissions for your Studio user profile execution role:
- managed policy [`AmazonSageMakerFullAccess`](https://us-east-1.console.aws.amazon.com/iamv2/home?region=us-east-1#/policies/details/arn%3Aaws%3Aiam%3A%3Aaws%3Apolicy%2FAmazonSageMakerFullAccess)
- `servicequotas:GetServiceQuota` API permission
- `bedrock:*` API permission

Use the following inline permission policy to add to the user profile execution role:
```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "ServiceQuotas",
            "Effect": "Allow",
            "Action": [
                "servicequotas:GetServiceQuota"
            ],
            "Resource": "*"
        },
        {
            "Sid": "Bedrock",
            "Effect": "Allow",
            "Action": "bedrock:*",
            "Resource": "*"
        }
    ]
}
```

Add `bedrock.amazonaws.com` to the user profile execution role trust relationships:
```json
{
    "Effect": "Allow",
    "Principal": {
        "Service": "bedrock.amazonaws.com"
    },
    "Action": "sts:AssumeRole"
}
```

#### Download source code into Amazon SageMaker Studio
In the Studio top menu select **File**, then **New**, then **Terminal**:

![](/static/img/studio-system-terminal-via-menu.png)

Run the following command in the terminal:
```
git clone https://github.com/aws-samples/generative-ai-on-aws-architecture-patterns.git
```

The code repository will be downloaded and saved in your home directory in Studio.

Now you all setup to start with implementation of your first generative AI design pattern.

## Generative AI design patterns
Before you start the hands-on activities on the workshop's design patterns, let's go through some theory.

While LLMs able to perform a broad range of tasks and can generalize or do complex reasoning well, they also lack domain-specific and up-to-date knowledge.

There are two broad foundational design pattern to connect an LLM to specialized skills or to up-to-date data:
1. **Parametric learning** – changing model's parameters/weights via gradient update/backpropagation by pre-training or fine-tuning of the model
2. **Non-parametric or in-context learning (ICL)** - using a model as is without any model update with its parameters/weights frozen by providing all relevant information and context via the input prompt

All design patterns in this workshop built on the in-context learning approach.

Each lab focuses on one of the following design patterns:

- [Lab 1](/content/lab-01/README.md): In-context learning and prompt engineering
- [Lab 2](/content/lab-02/README.md): Retrieval augmented generation (RAG) and conversational FAQ (CFAQ)
- Lab 3: Natural language queries (NLQ)
- Lab 4: Reasoning and react (ReAct)
- Lab 5: Multi-agents  

This set of design pattern is not exhaustive, but covers the majority of common and frequent use cases.

### In-context learning (ICL)
This very broad design pattern includes:
- Use LLMs off the shelf without fine-tuning
- Use natural language instructions in a prompt
- Provide relevant information in the prompt: instructions, context, examples, factual data, code, etc
- Do prompt engineering to improve model performance

It's important to understand, that ICL is not training and there is no parameter/weight update happening for a model.

The following example shows the ICL approach:

![](/static/img/icl-example.png)

Refer to the original paper [A Survey on In-context Learning](https://arxiv.org/abs/2301.00234) for more details.

## Get started with hands-on labs
Navigate to each lab's folder and follow the instructions in `README.md` file.

## Contributors
The baseline of source code and overall architecture were taken from the public AWS workshop [Implementing Generative AI on AWS](https://catalog.us-east-1.prod.workshops.aws/workshops/80ae1ed2-f415-4d3d-9eb0-e9118c147bd4/en-US).

The workshop authors:

[Yevgeniy Ilyin](https://www.linkedin.com/in/yevgeniyilyin/) | [Nikita Fedkin](https://www.linkedin.com/in/nikitafed/) |
:---:|:---:
![](/static/img/hyperbadge_ilyiny.png)  |  ![](/static/img/hyperbadge_nikitafe.png)


Special thanks to:
- [Mikael Mutafyan](https://www.linkedin.com/in/mikaelmutafyan/)
- [Aris Tsakpinis](https://www.linkedin.com/in/aris-tsakpinis-15a320143/)

for help with questions, recommendations, and the workshop review.

---

Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
SPDX-License-Identifier: MIT-0