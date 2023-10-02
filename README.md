# Generative AI retrieval-augmented generation (RAG) chatbot workshop

This **internal** repository contains instructions and examples to create your first generative AI chatbot connected to a knowledge base. 

In this workshop you're going to learn:
- a foundational generative AI design pattern: retrieval augmented generation (RAG)
- how to use Amazon Kendra to implement a knowledge base with a document index
- how to ingest documents into the knowledge base using Kendra connectors
- how to use Amazon SageMaker JumpStart to deploy an open source LLM as a real-time endpoint and use the endpoint for inference
- how to use HuggingFace TGI container and Amazon SageMaker to deploy an LLM as a real-time endpoint
- how to use Amazon Bedrock to connect to large language models (LLMs) and use them via an API
- how to use popular Python frameworks [streamlit](https://streamlit.io/) and [langchain](https://python.langchain.com/docs/get_started/introduction) to implement generative AI applications in hours

## Pre-requisites and environment setup
You need access to an AWS account. You can use your own isengard account or the shared isengard account, which is given by the workshop moderators.

If you use your own isengard account make sure to fulfill the following pre-requisites before the workshop:
1. Admin access to the account
2. Quota for `ml.g5.12xlarge` is set to at least 1. You can increase the quota in AWS console as described in [this instructions](https://aws.amazon.com/premiumsupport/knowledge-center/manage-service-limits/)
3. [AWS Console access isengard](https://isengard.amazon.com/console-access)

The next sections contain step-by-step instructions how to setup the required development environments.

### Clone the workshop repository
Clone the internal GitLab [repo](https://gitlab.aws.dev/ilyiny/genai-rag-bot-workshop) to your local notebook:

```sh
git clone git@ssh.gitlab.aws.dev:ilyiny/genai-rag-bot-workshop.git
```

### Setup AWS Cloud9 environment
- Sign in an AWS account 
- If you don't have an existing Cloud9 environment, setup a new one. See [Hello AWS Cloud9](https://docs.aws.amazon.com/cloud9/latest/user-guide/tutorial.html) for a step-by-step instruction
- Open the Cloud9 environment
- Upload the folder `genai-rag-bot-workshop` to the Cloud9 environment
- Open terminal in  Cloud9 and run `cd genai-rag-bot-workshop/`
- Run `aws sts get-caller-identity` and make sure you have the Administrator role
- Run `chmod +x resize-disk.sh`
- Run `./resize-disk.sh 100`

#### Upgrade to Python 3.10
The Lambda function you're going to implement requires Python 3.10 runtime. This in turn requires the Python 3.10 version of AWS SAM CLI. 

Follow the instructions below to upgrade Cloud9 Python version to 3.10:

---
```sh
sudo yum update -y
sudo yum erase openssl-devel -y
sudo yum install openssl11 openssl11-devel  libffi-devel bzip2-devel wget -y
wget https://www.python.org/ftp/python/3.10.2/Python-3.10.2.tgz
tar -xf Python-3.10.2.tgz
cd Python-3.10.2/
./configure --prefix=/usr --enable-optimizations
make -j $(nproc)
sudo make altinstall
python3.10 -V
```

```
sudo yum install python3-pip
```

```
sudo ln -sf /usr/bin/python3.10 /usr/bin/python3
```

```
source ~/.bash_profile
python --version
```

--- 
```
git clone https://github.com/pyenv/pyenv.git ~/.pyenv
cat << 'EOT' >> ~/.bashrc
export PATH="$HOME/.pyenv/bin:$PATH"
eval "$(pyenv init -)"
EOT
source ~/.bashrc
```

```
sudo yum update -y
sudo yum erase openssl-devel -y
sudo yum install openssl11 openssl11-devel xz-devel libffi-devel bzip2-devel wget -y
```

```
pyenv install 3.10
pyenv global 3.10
```

```
export PATH="$HOME/.pyenv/shims:$PATH"
```

```
source ~/.bash_profile
python --version
```
---

Upgrade SAM to the latest version:
```sh
sudo yum update -y
wget https://github.com/aws/aws-sam-cli/releases/latest/download/aws-sam-cli-linux-x86_64.zip
unzip aws-sam-cli-linux-x86_64.zip -d sam-installation
sudo ./sam-installation/install --update
sam --version
```

### Setup SageMaker Studio
SageMaker domain
User profile -> for each individual user

#### SageMaker execution role permissions
You need the following permissions for your Studio user profile execution role:
- `servicequotas:GetServiceQuota`
- `bedrock:*`

**TODO**: add all required permissions for the SageMaker execution role here
provide the inline policy json:
```json
```

**TODO**: attach the inline policy to the execution role

Add `bedrock.amazonaws.com` to the execution role trust relationships:
```json
{
    "Effect": "Allow",
    "Principal": {
        "Service": "bedrock.amazonaws.com"
    },
    "Action": "sts:AssumeRole"
}
```

Now you all setup to start with the RAG chatbot implementation.

## Generative AI design patterns
Before you start the implementation of the RAG chatbot, let's go through some theory.

While LLMs able to perform a broad range of tasks and can generalize or reason well, they also lack domain-specific and up-to-date knowledge.

There are two broad foundational design pattern to connect an LLM to a knowledge base.
1. **Parametric learning** – changing the model parameters/weights via gradient update/backpropagation by pre-training or fine-tuning model
2. **Non-parametric or in-context learning (ICL)** - using the model as is without any model update (parameters/weights are frozen) by providing all relevant information and context via the input prompt.

This workshop presents the in-context learning option, more specifically a foundational design pattern called retrieved augmented generation (RAG). Refer to the original paper [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401) for a technical overview.

There are the following popular ICL design patterns:
- RAG to connect an LLM to knowledge bases
- Natural language queries (NLQ) to connect an LLM to structured databases and to explore your databases with natural language questions
- [Reasoning and action (ReAct)](https://arxiv.org/abs/2210.03629)
- [Agents](https://go.aws/3EOgPaS)

This list is not exhaustive, but covers the majority of ICL use cases.

### In-context learning (ICL)
- Use LLMs off the shelf without fine-tuning
- Use natural language instructions in a prompt
- Provide relevant information in the prompt: instructions, context, examples, factual data, code, etc
- ICL is not training (no gradient update)

The following example shows the ICL approach:

![](./static/img/icl-example.png)

Refer to the original paper [A Survey on In-context Learning](https://arxiv.org/abs/2301.00234) for more details.

### RAG architecture
![](./static/img/rag-overview.png)

1. A user asks a question in the chatbot application
2. The chatbot sends the question to the retriever component
3. The retriever sends a search query to the knowledge base which is an information retrieval (IR) engine. The IR engine can be implemented using any technology. Refer to a deep dive on different IR approaches [How to Build an Open-Domain Question Answering System?](bit.ly/3ZppYAl) for more information. Very often the IR engine is a semantic search
4. The IR engine returns search results with document excerpts and links to relevant documents
5. Retriever sends the response to the chatbot/orchestrator
6. The chatbot/orchestrator send the user question concatenated with the search results as the LLM prompt. Here is important to understand the context length limitation and prompt engineering approaches for a specific LLM
7. The LLM summarize the information and uses the context in the prompt to provide a factual response
8. The chatbot sends the response back to the user

The following exhibit shows a RAG example:

![](./static/img/rag-example.png)

## Implementation
This section contains step-by-step instructions and all details needed to implement your first RAG-based generative AI application.

### Architecture overview
**TODO**: The real app architecture diagram (detailed)

![](static/design/rag-bot-architecture-overview.drawio.svg)

### Knowledge base
Navigate to the [Cloud9 environment](https://us-east-1.console.aws.amazon.com/cloud9control/home?region=us-east-1#/).

Create S3 bucket for documents
- `aws s3 mb s3://document-storage-<your alias>-<account-id>-<region>`

#### Amazon Kendra
[Amazon Kendra](https://docs.aws.amazon.com/kendra/latest/dg/what-is-kendra.html) is an enterprise search and uses semantic and contextual understanding capabilities to return relevant documents to a natural language search query.

You are going to use Amazon Kendra to implement a retriever part of the RAG chatbot application.

Create Kendra index:
1. Navigate to Kendra console
2. Choose **Create and index**
3. Provide a name for the index
4. Choose create a new role, provide the role name suffix, choose **Next**
6. Choose **No** for token access control, choose **Next** 
7. Choose **Developer edition**, choose **Next**
8. Review and choose **Create**

Wait until the Kendra index is created and ready.

**TODO**: add boto3/AWS CLI Kendra index creation 

#### Amazon OpenSearch
🚧 Available in the next version of the workshop!

#### Ingestion - Kendra
You're going to ingest public press releases from Swiss Government web site https://www.admin.ch/ using a built-in [Kendra Web Crawler connector 2.0](https://docs.aws.amazon.com/kendra/latest/dg/data-source-v2-web-crawler.html).

1. Navigate to the created index in the [Kendra console](https://us-east-1.console.aws.amazon.com/kendra/home?region=us-east-1#indexes) 
2. Choose **Add data source**
3. Choose **Web Crawler v2.0** connector and click **Add connector** You can choose any other connector to connect to a data source of your choice and ingest documents from that data source

The following instruction assumes you use the Web Crawler to ingest the documents from the site https://www.admin.ch/. If you use another data source or another web URL, configure the Kendra connector accordingly.

In the **Add data source** pane:

Step 1:
1. Provide a name for the data source, e.g. `admin-ch-public`, choose **Next**
2. Choose **Source URLs** and enter the following seed URLs:
```
https://www.admin.ch/gov/en/start/documentation/media-releases.html?dyn_pageIndex=0
https://www.admin.ch/gov/en/start/documentation/media-releases.html?dyn_pageIndex=1
https://www.admin.ch/gov/en/start/documentation/media-releases.html?dyn_pageIndex=2
https://www.admin.ch/gov/en/start/documentation/media-releases.html?dyn_pageIndex=3
https://www.admin.ch/gov/en/start/documentation/media-releases.html?dyn_pageIndex=4
https://www.admin.ch/gov/en/start/documentation/media-releases.html?dyn_pageIndex=5
https://www.admin.ch/gov/en/start/documentation/media-releases.html?dyn_pageIndex=6
https://www.admin.ch/gov/en/start/documentation/media-releases.html?dyn_pageIndex=7
https://www.admin.ch/gov/en/start/documentation/media-releases.html?dyn_pageIndex=8
https://www.admin.ch/gov/en/start/documentation/media-releases.html?dyn_pageIndex=9
```

These URLs contain 200 the most recent press releases in English.
    - Choose **No authentication**
    - Choose **Create a new role**, provide a role suffix, choose **Next**
3. Provide the following settings on **Configure sync settings** page:
    - Sync domains with subdomains only
    - Crawl depth: 1
    - Check **Include the files that has links to web pages**
    - Sync mode: Full sync
    - Frequency: Run on demand
4. Leave the default settings on **Set field mappings**
5. Review and create

Choose **Sync now**

The crawling and document indexing takes about 15 minutes. You don't need to wait until the sync finished and can move to the next task.

**TODO**: add boto3 or AWS CLI kendra index creation and ingestion

#### Ingestion - OpenSearch
🚧 Available in the next version of the workshop!

### Generator
You use an LLM as a generator to generate answers to the question using retrieved context.
Navigate to SageMaker Studio and open `notebooks/llm-generator.ipynb` notebook. Follow the instructions in the notebook to create an LLM real-time endpoint.

### Chatbot app
Navigate to the [Cloud9 environment](https://us-east-1.console.aws.amazon.com/cloud9control/home?region=us-east-1#/).

The chatbot application source code is in the folder [`chatbot`](./chatbot/) of the workshop. This folder contain the following files: 
- `app.py`: the frontend utilizing the popular streamlit framework
- `Dockerfile`: Dockerfile providing the blueprint for the creation of a Docker image
- `requirements.txt`: specifying the dependencies required to be installed for hosting the frontend application
- `setup.sh`: setup script consisting all the necessary steps to create a ECR repository, build the Docker image and push it to the respective repository you created

In Cloud9 terminal:
- `cd chatbot/`
- `bash setup.sh`

### Retriever and orchestration
The retriever in the RAG design pattern is responsible for sending search request to the knowledge base/information retrieval engine and retrieving search results.

You use [LangChain](https://python.langchain.com/en/latest/) framework to implement the retriever and also the orchestration layer

The main components for the LangChain-based orchestrator are:

`KendraIndexRetriever`
The implementation of Kendra retriever is in `orchestration/kendra` folder.

`ConversationalBufferWindowMemory`
    short-term memory -> this workshop
    long-term memory

`DynamoDBChatMessageHistory`
Since you use Lambda as a stateless serverless microservice for the orchestration layer, you use Amazon DynamoDB to persist conversation memory to a DynamoDB table. 

`LLM endpoint`
You use a SageMaker real-time endpoint created in the **Generator** section or Amazon Bedrock API.

`PromptTemplate`
LangChain provides prompt templates for specific use cases and LLMs.

`ConversationalRetrievalChain`
This chain has two steps. First, it condenses the current question and the chat history into a standalone prompt which is sent to the retriever. After the retrieving the search result sends the question and search results to a LLM. 
With the declarative nature of LangChain you can easily use a separate language model for each step. For example, you can use a cheaper and faster model for question summarization task, and a larger, more advanced and expensive model for answering the question. In this workshop you use one model for both steps.

To understand how the end-to-end orchestration works and how the components are linked together, look into the orchestration implementation in the Lambda function `orchestration/rag_app.py`.

**TODO**: Add Amazon Bedrock integration to Lambda

#### Orchestration layer deployment
Navigate to the [Cloud9 environment](https://us-east-1.console.aws.amazon.com/cloud9control/home?region=us-east-1#/).

You're going to use [AWS Serverless Application Model (AWS SAM)](https://docs.aws.amazon.com/serverless-application-model/latest/developerguide/what-is-sam.html) to deploy end-to-end RAG chatbot application.

The SAM CloudFormation template deploys the following resources:
- Network infrastructure including VPC, two subnets, and Internet Gateway
- IAM execution roles for AWS Lambda and ECS task
- ECS cluster for hosting the front-end
- Application Load Balancer for public access of the front-end
- Amazon API Gateway API exposing the orchestration layer to the frond-end via REST API
- AWS Lambda function with the orchestration layer implementation
- Amazon DynamoDB table for conversation history persistence

Look in `template.yaml` CloudFormation template and `orchestration/rag_app.py` Lambda function code to understand how the main components connected and how the serverless backend works.

Now deploy the SAM application.

Make sure you're in the workshop folder: 
```sh
cd ~/environment/genai-rag-bot-workshop/
```

Build AWS SAM application:
```bash
sam build
```

Deploy the application:
```bash
sam deploy --guided
```

You need to provide following parameters to pass to the SAM CloudFormation template:
- `LLMContextLength`
- `ECRImageURI`
- `KendraIndexId`
- `SageMakerLLMEndpointName`

Provide configuration parameters and wait until the CloudFormation stack deployment succeeded. 

Print the stack output (provide your stack name):
```sh
aws cloudformation describe-stacks \
    --stack-name rag-app  \
    --output table \
    --query "Stacks[0].Outputs[*].[OutputKey, OutputValue]"
```

Copy the value of `RAGChatBotUrl` in a browser and start the chatbot.

If everything works fine, you should see the chatbot user interface:

![](./static/img/rag-bot-ux.png)

## Experimentation
Now ask some questions about Switzerland or on generally any topic, for example:
`What is the usage of fossil fuels in Switzerland?`

## Conclusion
Congratulations, you just build your first RAG-based generative AI application on AWS!

## Clean up
If you use own AWS account or an isengard account, you must delete provisioning resources to avoid unnecessary charges. 

Remove the application CloudFormation stack:
- Execute in the Cloud9 terminal: `sam delete`. Wait until stacks are deleted

Remove the SageMaker endpoint:
- Navigate to SageMaker Studio
- Execute the **Clean up** section of the `llm-generator.ipynb` notebook

Delete the Cloud9 environment is you don't need it anymore

## Resources
The following is the collection of useful links to the related resources.

- [A Survey on In-context Learning](https://arxiv.org/abs/2301.00234) – a paper on in-context learning for LLMs
- [Deploy self-service question answering with the QnABot on AWS solution powered by Amazon Lex with Amazon Kendra and large language models](https://go.aws/48oG6WK) – in-depth example of RAG chatbot with Amazon Kendra
- [How to Build an Open-Domain Question Answering System?](https://bit.ly/3ZppYAl) – a good overview of information retrieval (IR) approaches
- [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401) – an original paper on RAG approach
- [Retrieval augmented generation (RAG)](https://docs.aws.amazon.com/sagemaker/latest/dg/jumpstart-foundation-models-customize-rag.html) – Amazon SageMaker Developer Guide
- [Implementing Generative AI on AWS workshop](https://catalog.us-east-1.prod.workshops.aws/workshops/80ae1ed2-f415-4d3d-9eb0-e9118c147bd4/en-US) – a public workshop
- [Large Language Model - Query Disambiguation for Conversational Retrieval, and Generative Question Answering](https://github.com/aws-solutions/qnabot-on-aws/tree/main/docs/LLM_Retrieval_and_generative_question_answering)
- [QnABot on AWS](https://aws.amazon.com/solutions/implementations/qnabot-on-aws/) - a public solution in AWS Solutions Library

## Contributors
The baseline of source code and overall architecture were taken from the public AWS workshop [Implementing Generative AI on AWS](https://catalog.us-east-1.prod.workshops.aws/workshops/80ae1ed2-f415-4d3d-9eb0-e9118c147bd4/en-US).

The workshop authors:

[Yevgeniy Ilyin](https://phonetool.amazon.com/users/ilyiny) | [Nikita Fedkin](https://phonetool.amazon.com/users/nikitafe) |
:---:|:---:
![](./static/img/hyperbadge_ilyiny.png)  |  ![](./static/img/hyperbadge_nikitafe.png)


Special thanks to:
- [Mikael Mutafyan](https://phonetool.amazon.com/users/mimuta)
- [Aris Tsakpinis](https://phonetool.amazon.com/users/tsaris)

for help with questions, recommendations, and workshop review.

---

Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
SPDX-License-Identifier: MIT-0