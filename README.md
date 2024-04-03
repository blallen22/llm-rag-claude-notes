# LLM - RAG with Claude for Graduate School Notes

This is a personal project that uses a Retrieval-Augmented Generation (RAG) Large Language Model (LLM) implementation to derive insights from my personal graduate school notes. My use case is that I enjoyed my graduate school experience but sometimes find myself wishing that I explicitly summarized the most important takeaways from each course. I want to be able to refresh my memory and generate insights across the 22 courses I took without having to read 1,000+ pages of text. This project allows me to solve that problem using Claude models (2 and Sonnet) via Amazon Bedrock, Amazon Titan Embeddings, and Meta's Facebook AI Similarity Search (FAISS) vector store to generate question-and-answer insights from my graduate school notes.

Disclaimer: This work is in a personal capacity unrelated to my employer. This package is for illustrative purposes and is not designed for end-to-end productionization as-is.

## Prerequisites

You will need your own Amazon Web Services (AWS) account with Claude and Titan Amazon Bedrock model access. Your Python environment will also require:
- langchain>=0.1.11
- langchain-community
- faiss-cpu==1.8.0

## Overview

This package will demonstrate how to:
- Import libraries
- Instantiate the LLM and embeddings models
- Load PDFs of notes as documents
- Split documents into chunks
- Confirm embeddings functionality
- Create vector store
- Define Claude 3 function
- Embed question and return relevant chunks
- Create prompt template
- Produce RAG outputs with Claude 2
- Produce outputs with Claude 3 Sonnet for comparison


## Claude 2 RAG Output Examples

When Claude 2 was provided with the vector store of user reviews and prompted "What are the most important concepts in Behavioral Economics?", it returned:

![image](https://github.com/blallen22/llm-rag-claude-notes/assets/4731381/0def7435-b6fb-46b8-bb61-3d3ed360b873)


When Claude 2 was provided with the vector store of user reviews and prompted "If I manage a small team that directly competes with larger, better-resourced teams, how can I more effectively compete against these teams?", it returned:

![image](https://github.com/blallen22/llm-rag-claude-notes/assets/4731381/2e9636a7-b445-453b-9109-b105f25057a9)


## Input Data

The input data for this project are my personal graduate school notes for 22 courses, which reflects thousands of pages of text. These word processing documents were converted to PDFs for easier loading for the vector store.

## Next Steps
Future next steps for this project include:
- Incorporating evaluation methodologies to assess the quality of the outputs beyond the current heuristic assessment
- Inspecting the methodological decisions with further granularity (e.g., the chunk size during the chunking process, etc.)
- Applying this approach to additional use cases

## License

This project is licensed under the [MIT License](https://choosealicense.com/licenses/mit/)

## Resources Referenced

  - [Amazon Bedrock Workshop - Langchain Knowledge Bases and RAG Examples](https://github.com/aws-samples/amazon-bedrock-workshop/blob/main/06_OpenSource_examples/01_Langchain_KnowledgeBases_and_RAG_examples/01_qa_w_rag_claude.ipynb)
  - [Pinecone Langchain RAG Examples](https://colab.research.google.com/github/pinecone-io/examples/blob/master/docs/langchain-retrieval-augmentation.ipynb)
