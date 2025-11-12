# GenAI-WebScraping-RAG-Chatbot
A Retrieval-Augmented Generation (RAG) chatbot that answers product-related queries using scraped e-commerce data, FAQs, and customer reviews. Built with Python, LangChain, OpenAI GPT, FAISS, and Flask, the system combines web scraping, semantic search, and LLM-powered generation to deliver context-aware, accurate, and conversational responses.

## Objective:
The goal of the ecommerce chatbot is to enhance the customer experience on the e-commerce platform by delivering fast, accurate, and personalized responses to their queries. The chatbot should:

Provide real-time product information and recommendations.
Retrieve relevant data from a knowledge base in response to customer questions.
Improve customer satisfaction through engaging conversations.
Assist in decision-making by answering product-related queries effectively.
How it Works
The ecommerce chatbot utilizes a combination of retrieval mechanisms and generative models to deliver accurate and context-aware responses to customer queries.

## Key Components:
Retrieval Mechanism: o The system maintains a pre-processed e-commerce knowledge base (product descriptions, FAQs, reviews, etc.). o When the user inputs a query, the retrieval component finds relevant documents or pieces of information related to the query from the knowledge base.
Generative Model: o After retrieving the relevant data, the generative model (such as a GPT-based model) is tasked with crafting a human-like response using the context provided by the retrieved documents.
Chatbot Pipeline: o Step 1: User submits a query (e.g., "What are the top features of Product X?"). o Step 2: The retrieval component searches for relevant documents or product details related to Product X from the knowledge base. o Step 3: The retrieved information is passed to the generative model, which formulates a coherent, context-aware response. o Step 4: The response is delivered to the user in a conversational manner.
## Architecture:
<img width="975" height="914" alt="1762961283153469919419124018016" src="https://github.com/user-attachments/assets/da7dd4c8-b1bd-475b-acf9-b8ea801bd22a" />

