#########################################
# LLM - RAG with Claude for Graduate School Notes
# Table of Contents
# 1. Import libraries
# 2. Instantiate the LLM and embeddings models
# 3. Load websites as documents
# 4. Split documents into chunks
# 5. Confirm embeddings functionality
# 6. Create vector store
# 7. Define Claude 3 function
# 8. Embed question and return relevant chunks
# 9. Create prompt template
# 10. Produce RAG outputs with Claude V2
# 11. Produce RAG outputs with Claude 3 Sonnet
#########################################





#########################################
# 1. Import libraries
import warnings
import json
import os
import sys
import boto3
import botocore
import textwrap
import numpy as np
import base64
import logging

from botocore.config import Config
from botocore.exceptions import ClientError
from IPython.display import display, Markdown, Latex
from io import StringIO
from typing import Optional
from langchain.embeddings import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.document_loaders.pdf import PyPDFLoader, PyPDFDirectoryLoader
from langchain.chains.question_answering import load_qa_chain
from langchain.vectorstores import FAISS
from langchain.indexes import VectorstoreIndexCreator
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

warnings.filterwarnings('ignore')
#########################################





#########################################
# 2. Instantiate the LLM and embeddings models
boto3_bedrock = boto3.client('bedrock-runtime')

# Claude V2
llm = Bedrock(model_id="anthropic.claude-v2", client=boto3_bedrock, model_kwargs={'max_tokens_to_sample':200})

# Titan Embeddings
bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client=boto3_bedrock)
#########################################





#########################################
# 3. Load PDFs of notes as documents
loader = PyPDFDirectoryLoader("./data/")
documents = loader.load()
#########################################





#########################################
# 4. Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap  = 100,
)
docs = text_splitter.split_documents(documents)

# print statistics about the documents
avg_doc_length = lambda documents: sum([len(doc.page_content) for doc in documents])//len(documents)
avg_char_count_pre = avg_doc_length(documents)
avg_char_count_post = avg_doc_length(docs)
print(f'Average length among {len(documents)} documents loaded is {avg_char_count_pre} characters.')
print(f'After the split we have {len(docs)} documents more than the original {len(documents)}.')
print(f'Average length among {len(docs)} documents (after split) is {avg_char_count_post} characters.')
#########################################





#########################################
# 5. Confirm embeddings functionality
try:
    sample_embedding = np.array(bedrock_embeddings.embed_query(docs[0].page_content))
    print("Sample embedding of a document chunk: ", sample_embedding)
    print("Size of the embedding: ", sample_embedding.shape)

except ValueError as error:
    if "AccessDeniedException" in str(error):
        print(f"\x1b[41m{error}\
        \nTo troubeshoot this issue please refer to the following resources.\
         \nhttps://docs.aws.amazon.com/IAM/latest/UserGuide/troubleshoot_access-denied.html\
         \nhttps://docs.aws.amazon.com/bedrock/latest/userguide/security-iam.html\x1b[0m\n")
        class StopExecution(ValueError):
            def _render_traceback_(self):
                pass
        raise StopExecution
    else:
        raise error
#########################################





#########################################
# 6. Create vector store
vectorstore_faiss = FAISS.from_documents(
    docs,
    bedrock_embeddings,
)

wrapper_store_faiss = VectorStoreIndexWrapper(vectorstore=vectorstore_faiss)
#########################################





#########################################
# 7. Define Claude 3 function
logger = logging.getLogger(__name__)

def claude_3_text(prompt):
    # Initialize the Amazon Bedrock runtime client
    client = boto3.client(
        service_name="bedrock-runtime", region_name="us-east-1"
    )

    # Invoke Claude 3 Sonnet
    model_id = "anthropic.claude-3-sonnet-20240229-v1:0"

    try:
        response = client.invoke_model(
            modelId=model_id,
            body=json.dumps(
                {
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": 1024,
                    "messages": [
                        {
                            "role": "user",
                            "content": [{"type": "text", "text": prompt}],
                        }
                    ],
                }
            ),
        )

        # Process and print the response
        result = json.loads(response.get("body").read())
        output_list = result.get("content", [])

        for output in output_list:
            print(output["text"])

        return result
    except ClientError as err:
        logger.error(
            "Couldn't invoke Claude 3 Sonnet. Here's why: %s: %s",
            err.response["Error"]["Code"],
            err.response["Error"]["Message"],
        )
        raise
#########################################





#########################################
# 8. Embed question and return relevant chunks
def print_ww(*args, width: int = 100, **kwargs):
    """Like print(), but wraps output to `width` characters (default 100)"""
    buffer = StringIO()
    try:
        _stdout = sys.stdout
        sys.stdout = buffer
        print(*args, **kwargs)
        output = buffer.getvalue()
    finally:
        sys.stdout = _stdout
    for line in output.splitlines():
        print("\n".join(textwrap.wrap(line, width=width)))

query = """What are the most important concepts in Behavioral Economics?"""

query_embedding = vectorstore_faiss.embedding_function.embed_query(query)
np.array(query_embedding)

relevant_documents = vectorstore_faiss.similarity_search_by_vector(query_embedding)
print(f'{len(relevant_documents)} documents are fetched which are relevant to the query.')
print('----')
for i, rel_doc in enumerate(relevant_documents):
    print_ww(f'## Document {i+1}: {rel_doc.page_content}.......')
    print('---')
#########################################





#########################################
# 9. Create prompt template
prompt_template = """

Human: Use the following pieces of context to provide a concise answer to the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.
<context>
{context}
</context

Question: {question}

Assistant:"""

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)
#########################################





#########################################
# 10. Produce RAG outputs with Claude V2
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore_faiss.as_retriever(
        search_type="similarity", search_kwargs={"k": 3}
    ),
    return_source_documents=True,
    chain_type_kwargs={"prompt": PROMPT}
)
answer = qa({"query": query})
print_ww(answer)
display(Markdown(answer['result']))
#########################################





#########################################
# 11. Produce RAG outputs with Claude 3 Sonnet
text_prompt = "What are the most important concepts in Behavioral Economics?"
claude_3_text(text_prompt)
#########################################
