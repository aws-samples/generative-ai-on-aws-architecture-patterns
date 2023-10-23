# Retrieval augmented generation chat bot
This folder contains the source code for RAG chat bot using Amazon Kendra and Amazon Bedrock.

## RAG architecture
RAG design pattern is an extension of ICL where you connect a model to a knowledge base. Refer to the original paper [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401) for a technical overview.

![](/static/img/rag-overview.png)

1. A user asks a question in a chatbot application
2. The chatbot sends the question to a retriever component
3. The retriever prepares and sends a search query to a knowledge base which is an information retrieval (IR) engine. The IR engine can be implemented using any technology. Refer to a deep dive on different IR approaches [How to Build an Open-Domain Question Answering System?](https://bit.ly/3ZppYAl) for more information. Very often the IR engine is a semantic search
4. The IR engine returns search results with document excerpts and links to relevant documents
5. The retriever sends the response to the chatbot/orchestrator
6. The chatbot/orchestrator send the user question concatenated with the search results as an LLM prompt to an LLM of your choice. Here is important to understand the context length limitation and prompt engineering approaches for a specific LLM
7. The LLM summarize the information and uses the context in the prompt to provide a factual response
8. The chatbot sends the response back to the user

The following exhibit shows a RAG example:

![](/static/img/rag-example.png)

## Implementation
This section contains step-by-step instructions and all details needed to implement your first RAG-based generative AI application.

### Architecture overview
Aimed with the theoretical knowledge, you're about to implement the following architecture. You use AWS services as building blocks to implement a scalable, secure, and reliable solution.

![](/static/design/rag-bot-architecture-overview.drawio.svg)

### Knowledge base
In this section you're going to create and populate a knowledge base you're going to connect to the chatbot.

Navigate to the [AWS Cloud9 environment](https://us-east-1.console.aws.amazon.com/cloud9control/home?region=us-east-1#/).

If you'd like to ingest some documents from an Amazon S3 bucket, you can create a dedicated bucket to be connected to Amazon Kendra:
- `aws s3 mb s3://document-storage-<your alias>-<account-id>-<region>`

#### Amazon Kendra
[Amazon Kendra](https://docs.aws.amazon.com/kendra/latest/dg/what-is-kendra.html) is an enterprise search service and uses semantic and contextual understanding capabilities to return relevant documents to a natural language search query.

In this workshop you are going to use Amazon Kendra to implement a retriever part of the RAG chatbot application.

Create Kendra index:
1. Navigate to [Amazon Kendra console](https://us-east-1.console.aws.amazon.com/kendra/home?region=us-east-1#/)
2. Choose **Create and index**
![](/static/img/kendra-create-index.png)
3. Provide a name for the index
4. Choose create a new role, provide the role name suffix, choose **Next**
![](/static/img/kendra-index-details.png)
6. Choose **No** for token access control, choose **Next** 
![](/static/img/kendra-user-access-control.png)
7. Choose **Developer edition**, choose **Next**
![](/static/img/kendra-choose-edition.png)
8. Review and choose **Create**

Wait until the Amazon Kendra index is created and ready:

![](/static/img/kendra-index.png)

**TODO**: add boto3/AWS CLI Kendra index creation 

#### Amazon OpenSearch
🚧 Available in the next version of the workshop!

#### Ingestion - Amazon Kendra
You're going to ingest public press releases from Swiss Government web site https://www.admin.ch/ using a built-in [Amazon Kendra Web Crawler connector 2.0](https://docs.aws.amazon.com/kendra/latest/dg/data-source-v2-web-crawler.html).

To create web crawler and ingest documents to the index:
1. Navigate to the created index in the [Amazon Kendra console](https://us-east-1.console.aws.amazon.com/kendra/home?region=us-east-1#indexes) 
2. Choose **Add data source**
![](/static/img/kendra-add-data-source.png)
3. Choose **Web Crawler v2.0** connector and click **Add connector** You can choose any other connector to connect to a data source of your choice and ingest documents from that data source
![](/static/img/kendra-web-crawler.png)

The following instruction assumes you use the Web Crawler to ingest the documents from the site https://www.admin.ch/. If you use another data source or another web URL, configure the Amazon Kendra connector accordingly.

In the **Add data source** pane:

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
These URLs contain about 200 the most recent press releases in English.  
    - Choose **No authentication**  
    - Choose **Create a new role**, provide a role suffix, choose **Next**  
3. Provide the following settings on **Configure sync settings** page:  
    - Sync domains with subdomains only  
    - Crawl depth: **1**  
    - Check **Include the files that has links to web pages**  
    - Sync mode: **Full sync**  
    - Frequency: **Run on demand**  
4. Leave the default settings on **Set field mappings**  
5. Review and create

Choose **Sync now**
![](/static/img/kendra-sync-data-source.png)

The crawling and document indexing takes about 15 minutes. You don't need to wait until the sync finished and can move to the next task.

**TODO**: add boto3 or AWS CLI kendra index creation and ingestion

#### Ingestion - OpenSearch
🚧 Available in the next version of the workshop!

### Generator
You use an LLM as a generator to generate answers to the question using retrieved context.
Navigate to SageMaker Studio and open `content/lab-02/notebooks/llm-generator.ipynb` notebook. Follow the instructions in the notebook to create an LLM real-time endpoint.

The deployment of an LLM real-time endpoint takes about 15 minutes.

### Chatbot app
In this section you create a front-end app container to run on [AWS Fargate](https://docs.aws.amazon.com/AmazonECS/latest/userguide/what-is-fargate.html).

Navigate to the [Cloud9 environment](https://us-east-1.console.aws.amazon.com/cloud9control/home?region=us-east-1#/).

The chatbot application source code is in the folder [`chatbot`](./chatbot/) of the workshop. This folder contain the following files: 
- `app.py`: the frontend utilizing the popular [streamlit](https://streamlit.io/) framework
- `Dockerfile`: Dockerfile providing a script for creation of a Docker image
- `requirements.txt`: specifies dependencies required to be installed for hosting the frontend application
- `setup.sh`: setup script consisting all the necessary steps to create a ECR repository, build the Docker image and push it to the respective repository you created

In Cloud9 terminal:
- `cd ~/environment/generative-ai-on-aws-architecture-patterns/content/lab-02/chatbot/`
- `bash setup.sh`

Make sure the container build and push finished successfully and the new image has been pushed to the ECR.

### Retriever and orchestration
The final and the most complex part of the application is to connect all components, such as LLM, UX, and retriever, together and implement orchestration of the data flow.

The retriever in the RAG design pattern is responsible for sending search request to the knowledge base or information retrieval engine and retrieving search results.

You use [LangChain](https://python.langchain.com/en/latest/) framework to implement the retriever and also the orchestration layer

The main components for the LangChain-based orchestrator are:

`AmazonKendraRetriever`  
You use a built-in Amazon Kendra retriever in LangChain. This class provides an abstraction of a retriever component and allows LangChain to interact with Amazon Kendra as part of conversation chain.

You also have a custom implementation of a Amazon Kendra retriever, `KendraIndexRetriever` class in the `orchestration/kendra` folder. This class is not used in the workshop. The implementation for your reference, you can try to use own retriever for any specific requirements.

`ConversationalBufferWindowMemory`  
This built-in LangChain class implements chat memory. There are two types of memory:
- short-term memory: related to one chain of a conversation. The workshop uses this type of memory
- long-term memory: related to all conversations between a user and a model. Long-term memory is useful for data analytics, validation, and model fine-tuning.

`DynamoDBChatMessageHistory`  
Since you use Lambda as a stateless serverless microservice for the orchestration layer, you use Amazon DynamoDB to persist conversation memory to a DynamoDB table. The workshop also uses a built-in LangChain class to minimize the implementation effort.

`LLM endpoint`  
You use a SageMaker real-time endpoint created in the **Generator** section or Amazon Bedrock API. The workshop uses a built-in LangChain class to abstract an LLM.

`PromptTemplate`  
LangChain provides prompt templates for specific use cases and LLMs.

`ConversationalRetrievalChain`  
Here the workshop again uses the existing LangChain class to implement a more complex flow for multi-hop conversation between an LLM, a user, and a retriever.

The conversational chain has two steps. 
1. The chain condenses the current question and the chat history into a standalone prompt which is sent to the retriever
2. After the retrieving the search result sends the question and search results to a LLM. 

With the declarative nature of LangChain you can easily use a separate language model for each step. For example, you can use a cheaper and faster model for question summarization task, and a larger, more advanced and expensive model for answering the question. In this workshop you use one model for both steps.

To understand how the end-to-end orchestration works and how the components are linked together, look into the orchestration implementation in the Lambda function `content/lab-02/orchestration/rag_app.py`.

#### Orchestration layer deployment
In this section you're going to deploy the end-to-end application stack, including UX, the backend API, and the serverless orchestration layer implemented as a Lambda function.

Navigate to the [AWS Cloud9 environment](https://us-east-1.console.aws.amazon.com/cloud9control/home?region=us-east-1#/).

You're going to use [AWS Serverless Application Model (AWS SAM)](https://docs.aws.amazon.com/serverless-application-model/latest/developerguide/what-is-sam.html) to deploy the RAG chatbot application.

The SAM CloudFormation template deploys the following resources:
- Network infrastructure including VPC, two public subnets, and an Internet Gateway
- IAM execution roles for AWS Lambda and ECS task
- ECS cluster for hosting the front-end
- Application Load Balancer for public access of the front-end
- Amazon API Gateway API exposing the orchestration layer to the frond-end via REST API
- AWS Lambda function with the orchestration layer implementation
- Amazon DynamoDB table for conversation history persistence

Look in `template.yaml` CloudFormation template and `content/lab-02/orchestration/rag_app.py` Lambda function code to understand how the main components connected and how the serverless backend works.

Now deploy the SAM application.

Make sure you're in the workshop folder: 
```sh
cd ~/environment/generative-ai-on-aws-architecture-patterns/content/lab-02/
```

Build AWS SAM application:
```sh
sam build
```

Deploy the application:
```sh
sam deploy --guided
```

You need to provide following parameters to pass to the SAM CloudFormation template:
- `LLMContextLength`: use default 2048 if you use Falcon 40B endpoint otherwise set accordingly to your LLM of choice
- `ECRImageURI`: use the ECR URI for `rag-app` image you built in the **Chatbot app** step
- `KendraIndexId`: use the id of the Amazon Kendra index you created
- `SageMakerLLMEndpointName`: use the endpoint name you created if you use a SageMaker endpoint, otherwise leave it empty
- `VPCCIDR`: CIDR block for the VPC, you can leave default if there is no conflicts with existing VPCs in your account

Please note, if you don't deploy a SageMaker LLM endpoint, you can use Amazon Bedrock API only.

Provide configuration parameters and wait until the CloudFormation stack deployment succeeded. 

Print the stack output (provide your stack name):
```sh
aws cloudformation describe-stacks \
    --stack-name <sam stack name>  \
    --output table \
    --query "Stacks[0].Outputs[*].[OutputKey, OutputValue]"
```

Copy the value of `RAGChatBotUrl` in a browser and start the chatbot.

If everything works fine, you should see the chatbot user interface:

![](/static/img/rag-bot-ux.png)

## Experimentation
Now ask some questions about Switzerland or on generally any topic, for example:
`What is the usage of fossil fuels in Switzerland?` or `What is the inflation forecast in Switzerland in 2023?`.

![](/static/img/rag-bot-ux-conversation.png)

### Try out various Amazon Bedrock LLMs in console
Optional activity, time permits:

- Use Amazon Kendra search functionality in the console
- Use Amazon Bedrock playground to try:
    - zero-shot prompt without search context
    - zero-shot prompt with search context
    - Engineered prompt with search context. For a prompt example see [here](https://smith.langchain.com/hub/hwchase17/weblangchain-generation)
    - Conversational chain with search context

## Conclusion
Congratulations, you just build your first RAG-based generative AI application on AWS!

## Clean up
If you use own AWS account, you must delete provisioning resources to avoid unnecessary charges. You don't need to clean up if you use a workshop instructor provided account.

Remove the application CloudFormation stack:
- Execute in the Cloud9 terminal: `sam delete`. Wait until stacks are deleted

If you used a SageMaker LLM endpoint, remove it:
- Navigate to SageMaker Studio
- Execute the **Clean up** section of the `llm-generator.ipynb` notebook

Delete the AWS Cloud9 environment is you don't need it anymore.

Delete the Amazon Kendra data source and Amazon Kendra index.

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
- [Building (and Breaking) WebLangChain](https://blog.langchain.dev/weblangchain/)

---
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
SPDX-License-Identifier: MIT-0

