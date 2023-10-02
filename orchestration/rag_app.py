import boto3
import json
import os

from langchain.chains import ConversationalRetrievalChain
from langchain.llms import SagemakerEndpoint
from langchain.prompts import PromptTemplate
from langchain.embeddings import SagemakerEndpointEmbeddings
from langchain.embeddings.sagemaker_endpoint import EmbeddingsContentHandler
from langchain.llms.sagemaker_endpoint import ContentHandlerBase, LLMContentHandler
from langchain.memory import ConversationBufferWindowMemory
from langchain.llms.bedrock import Bedrock
from langchain.memory.chat_message_histories import DynamoDBChatMessageHistory
from langchain.retrievers import AmazonKendraRetriever


REGION = os.environ.get('REGION')
KENDRA_INDEX_ID = os.environ.get('KENDRA_INDEX_ID')
SM_ENDPOINT_NAME = os.environ.get('SM_ENDPOINT_NAME')
LLM_CONTEXT_LENGTH = os.environ.get('LLM_CONTEXT_LENGTH', '2048')
USE_BEDROCK = os.environ.get('USE_BEDROCK', "False")

# Generative LLM 

# Content Handler for Falcon40b-instruct - please uncomment below if you used this option
class ContentHandler(LLMContentHandler):
    content_type = "application/json"
    accepts = "application/json"

    def transform_input(self, prompt, model_kwargs):
        input_str = json.dumps(
            {
                "inputs": prompt,
                "parameters": 
                {
                    "do_sample": False,
                    "repetition_penalty": 1.1, 
                    "return_full_text": False, 
                    "max_new_tokens": 1024
                }
            }
        )
        return input_str.encode('utf-8')
    
    def transform_output(self, output):
        response_json = json.loads(output.read().decode("utf-8"))
        return response_json[0]["generated_text"]

content_handler = ContentHandler()
    
_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language. 

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""

CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)

# retriever.get_relevant_documents(query)
if USE_BEDROCK == 'True':
    llm = Bedrock(client=boto3.client(service_name='bedrock'),
                model_id='anthropic.claude-v2'
    )
else:
    # SageMaker langchain integration, to assist invoking SageMaker endpoint.
    llm = SagemakerEndpoint(
        endpoint_name=SM_ENDPOINT_NAME,
        region_name=REGION,
        # model_kwargs={}
        content_handler=content_handler, 
    )

def lambda_handler(event, context):

    print(event)

    body = json.loads(event['body'])
    print(body)

    query = body['query']
    uuid = body['uuid']
    print(query)
    print(uuid)

    message_history = DynamoDBChatMessageHistory(
        table_name="MemoryTable", 
        session_id=uuid
    )
    memory = ConversationBufferWindowMemory(
        memory_key="chat_history",
        chat_memory=message_history, 
        return_messages=True, 
        k=3
    )

    # This retriever is using the new Kendra retrieve API https://aws.amazon.com/blogs/machine-learning/quickly-build-high-accuracy-generative-ai-applications-on-enterprise-data-using-amazon-kendra-langchain-and-large-language-models/
    retriever = AmazonKendraRetriever(
        index_id=KENDRA_INDEX_ID,
        region_name=REGION,
        top_k=2
    )
    
    qa = ConversationalRetrievalChain.from_llm(
        llm=llm, 
        retriever=retriever, 
        memory=memory, 
        condense_question_prompt=CONDENSE_QUESTION_PROMPT, 
        verbose=True
    )

    response = qa.run(query)   
    clean_response = response.replace('\n','').strip()

    print(clean_response)

    return {
        'statusCode': 200,
        'body': json.dumps(f'{clean_response}')
    }
