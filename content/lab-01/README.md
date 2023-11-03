# In-context-learning and prompt engineering
This folder contains the source code for a generative AI chatbot using Amazon Bedrock.

## Chatbot architecture

In this lab you're going to try the most common approaches to use an LLM on AWS:
1. Use Amazon Bedrock
2. Use Amazon SageMaker JumpStart
3. Use Amazon SageMaker real-time and asynchronous endpoints

You're also going to implement a simple generative AI conversational chatbot - a fully functional application with a front-end, a backend API, backend implementation, and a chat history persistent storage.

You use the following serverless architecture to implement these application layers:

![](/static/design/icl-architecture-aws.drawio.svg)

**Front-end**:  
- [Elastic Load Balancing (ELB)](https://docs.aws.amazon.com/elasticloadbalancing/latest/application/introduction.html) for scalable application access
- [Amazon Elastic Container Service (ECS)](https://docs.aws.amazon.com/AmazonECS/latest/developerguide/Welcome.html) for hosting the front-end container

**Backend API**:  
- [Amazon API Gateway](https://docs.aws.amazon.com/apigateway/latest/developerguide/welcome.html) for implementing, exposing, and managing API

**Backend**:
- [AWS Lambda](https://docs.aws.amazon.com/lambda/latest/dg/welcome.html) for backend implementation and orchestration layer
- [Amazon Dynamo DB](https://docs.aws.amazon.com/amazondynamodb/latest/developerguide/Introduction.html) for chat history persistent storage

Additionally, you're going to use [LangChain](https://www.langchain.com/) to simplify communication with LLM and to manage orchestration, and [streamlit](https://streamlit.io/) to implement the front-end.

**LLM**
- [Amazon Bedrock](https://docs.aws.amazon.com/bedrock/latest/userguide/what-is-service.html) for API-based access to LLMs
- [Amazon SageMaker](https://docs.aws.amazon.com/sagemaker/latest/dg/realtime-endpoints.html) for an LLM self-hosting option

### LLM endpoint
Create an LLM endpoint only if you'd like to experiment with SageMaker endpoints and JumpStart, otherwise move to **Front-end** section.

In this section you create a SageMaker LLM endpoint and you can experiment with real-time or asynchronous inference options.

Navigate to SageMaker Studio and open `content/notebooks/llm-endpoints.ipynb` notebook. Follow the instructions in the notebook to create an LLM endpoint hosted on SageMaker.

The deployment of an LLM endpoint takes about 15 minutes.

### Front-end
In this section you create a front-end app container to run on [AWS Fargate](https://docs.aws.amazon.com/AmazonECS/latest/userguide/what-is-fargate.html).

Navigate to the [AWS Cloud9 environment](https://us-east-1.console.aws.amazon.com/cloud9control/home?region=us-east-1#/).

The chatbot application source code is in the folder `content/lab-01/chatbot` of the workshop. This folder contain the following files: 
- `app.py`: the frontend utilizing the popular [streamlit](https://streamlit.io/) framework
- `Dockerfile`: Dockerfile providing a script for creation of a Docker image
- `requirements.txt`: specifies dependencies required to be installed for hosting the frontend application
- `setup.sh`: setup script consisting all the necessary steps to create a ECR repository, build the Docker image and push it to the respective repository you created

In the AWS Cloud9 terminal run the following commands:
```sh
cd ~/environment/generative-ai-on-aws-architecture-patterns/content/lab-01/chatbot/
bash setup.sh
```

To understand how the chatbot front-end works, look into the `app.py` file. The front-end uses the backend API to call the LLM:

```sh
def generate_response(prompt):
    url = f'{base_url}/backendapp'
    use_bedrock = st.session_state['USE_BEDROCK']
    body = {"query": prompt, "uuid": session_id, "USE_BEDROCK": use_bedrock}
    response = requests.post(url, headers=headers, data=json.dumps(body), verify=False)
    output_text = response.text
    return output_text
```

### API and backend

#### Abstract LLM via LangChain
[LangChain](https://python.langchain.com/docs/get_started/introduction) is a framework to develop generative AI application and use LLMs. It implements the main design patterns and provides lot of useful abstraction classes. You can implement your generative AI proof-of-concept or a productive application fast and with minimal effort.

This workshop uses LangChain to abstract complexity of using LLMs for many use cases, such as conversation, prompt engineering and augmentation, agents, or moderation.

For this lab you use almost the minimal setup to implement a conversational chat with memory. 
Look into the Lambda function code in the file `content/lab-01/backend/backend-app/backend_app.py` to understand how the code uses LangChain classes and accesses the LLM:

```python
llm = get_llm(use_bedrock)

message_history = DynamoDBChatMessageHistory(
    table_name=DDB_MEMORY_TABLE,
    session_id=uuid
)

memory = ConversationBufferWindowMemory(
    chat_memory=message_history,
    return_messages=True,
    k=int(CHAT_WINDOW_SIZE),
)

chain = ConversationChain(
    llm=llm,
    verbose=True,
    memory=memory,
)

response = chain.run(query)
```

The implementation uses the following LangChain classes:

`DynamoDBChatMessageHistory`  
This class implements an interface to Amazon DynamoDB to store and retrieve chat history. You only need to specify a DynamoDB table name and primary key used to keep history for a chat session.

`ConversationBufferWindowMemory`  
This class keeps the chat history but uses only the last `K` interactions to limit the prompt length. You can consider this class as a sliding window of the most recent interactions with LLM.

`ConversationChain`  
This class implements all needed structure and orchestration to interact with an LLM. It uses a specific prompt template, constructs a prompt based on the user input and chat history, sends the prompt to an LLM, and receives the answer. 

Feel free to experiment with other LangChain classes, constructs, and use cases following the LangChain [use cases](https://python.langchain.com/docs/use_cases/qa_structured/sql).

#### Backend deployment
In this section you're going to deploy the end-to-end application stack, including UX, the backend API, and the backend implemented as a Lambda function.

Navigate to the [AWS Cloud9 environment](https://us-east-1.console.aws.amazon.com/cloud9control/home?region=us-east-1#/).

You're going to use [AWS Serverless Application Model (AWS SAM)](https://docs.aws.amazon.com/serverless-application-model/latest/developerguide/what-is-sam.html) to deploy the LLM chatbot application.

The SAM CloudFormation template deploys the following resources:
- IAM execution roles for AWS Lambda and Amazon ECS task
- Amazon ECS cluster for hosting the front-end
- Application Load Balancer for public access of the front-end via HTTPS
- Amazon API Gateway API exposing the backend to the frond-end via REST API
- AWS Lambda function with the backend implementation
- Amazon DynamoDB table for chat history persistence

Look in `content/lab-01/template.yaml` CloudFormation template to understand what AWS services are used and how the main components connected and configured.

Now deploy the SAM application.

Make sure you're in the workshop folder:  
`cd ~/environment/generative-ai-on-aws-architecture-patterns/`

Change to the `content/lab-01` folder:  
`cd content/lab-01/`

Build AWS SAM application:  
`sam build`

Deploy the application:  
`sam deploy --guided`

You need to provide following parameters to pass to the SAM CloudFormation template:
- `LLMContextLength`: use default 2048 if you use Falcon 40B endpoint otherwise set accordingly to your LLM of choice
- `ECRImageURI`: use the ECR URI for `chatbot-app` image you built in the **Chatbot app** step
- `SageMakerLLMEndpointName`: use the endpoint name you created if you use a SageMaker LLM endpoint, otherwise leave empty

Answer `Y` to both prompts `Allow SAM CLI IAM role creation [Y/n]:` and `BackendAppFunction has no authentication. Is this okay? [y/N]`.

Provide configuration parameters and wait until the CloudFormation stack deployment succeeded. 

The deployment of the chatbot application takes about 10 minutes.

Print the stack output (provide your stack name):
```sh
aws cloudformation describe-stacks \
    --stack-name <your stack name> \
    --output table \
    --query "Stacks[0].Outputs[*].[OutputKey, OutputValue]"
```

Copy the value of `LLMChatBotUrl` in a browser and start the chatbot.

If everything works fine, you should see the chatbot user interface:

![](/static/img/chat-bot-ux.png)

## Clean up
If you use own AWS account, you must delete provisioning resources to avoid unnecessary charges. You don't need to clean up if you use a workshop instructor provided account.

Remove the application CloudFormation stack:
- Execute in the AWS Cloud9 terminal:
```sh
cd ~/environment/generative-ai-on-aws-architecture-patterns/content/lab-01/
sam delete
```

Wait until stacks are deleted.

If you used a SageMaker LLM endpoint, remove it:
- Navigate to SageMaker Studio
- Execute the **Clean up** section of the `content/notebooks/llm-endpoints.ipynb` notebook

Delete the AWS Cloud9 environment is you don't need it anymore.

## Resources
The following is the collection of useful links to the related resources.

- [A Survey on In-context Learning](https://arxiv.org/abs/2301.00234)
- [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165)
- [Large Language Models Are Human-Level Prompt Engineers](https://arxiv.org/abs/2211.01910)

Prompt engineering guides:
- [Prompt engineering guide](https://www.promptingguide.ai/)
- [AI21 Prompt Engineering](https://docs.ai21.com/docs/prompt-engineering) 
- [Anthropic Claude Prompt Design](https://docs.anthropic.com/claude/docs/introduction-to-prompt-design) 
- [Cohere Prompt Engineering](https://docs.cohere.com/docs/prompt-engineering)

---
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
SPDX-License-Identifier: MIT-0

