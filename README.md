Customer Service AI Chatbot

A modular customer service chatbot built with Python and Streamlit. The project combines a task-based customer service workflow with a Retrieval-Augmented Generation (RAG) knowledge base to support accurate and structured responses.

1. Project Overview

The system processes customer messages through a unified runtime and integrates security, natural-language processing, intent handling, ticket management, priority processing, routing, and knowledge-base retrieval.

Customer Message
       │
       ▼
Security & Validation
       │
       ▼
Conversation / NLU Processing
       │
       ▼
Intent & Sentiment Analysis
       │
       ▼
Ticket / Priority / Routing
       │
       ▼
Knowledge Base Retrieval
       │
       ▼
Customer Response

The application is provided through a Streamlit interface, while the runtime layer connects the individual task modules.

2. Key Features

Interactive customer service chat interface

RAG-based knowledge retrieval

CSV-based knowledge source

Modular task-based architecture

Conversation and session handling

Security validation

Natural-language and intent processing

Sentiment and risk detection

Ticket and priority management

Routing and escalation support

Knowledge-base version management

Runtime logging and monitoring

3. Project Structure

Customer_Service_Chatbot/
│
├── app.py
├── main.py
├── runtime.py
├── langchain llm.py
├── requirements.txt
│
├── dataset/
│   └── dataset.csv
│
├── runtime_data/
│   └── Knowledge-base runtime data
│
├── task 1/
├── task 2/
├── task 3/
├── task 4/
├── task 5/
└── task 6/

Core Components

app.py
Provides the Streamlit-based chatbot interface.

runtime.py
Acts as the main integration layer and connects the chatbot interface with the existing task modules and knowledge-base workflow.

dataset/dataset.csv
Provides the customer-service knowledge used by the retrieval system.

task 1 – task 6
Contain the individual processing modules responsible for the different customer-service functions.

runtime_data/
Contains generated knowledge-base and runtime information.

4. Dataset Integration

The chatbot uses dataset/dataset.csv as its knowledge source. The runtime processes the dataset through the knowledge-base pipeline and makes the resulting information available to the retrieval system.

When the dataset is updated, the knowledge base can be refreshed so that new or modified customer-service information becomes available to the chatbot.

5. Installation

Requirements

Python 3.10+

VS Code or another Python IDE

pip

Internet connection for installing dependencies and any configured external AI services

Setup

Create a virtual environment:

python -m venv venv

Windows:

venv\Scripts\activate

macOS/Linux:

source venv/bin/activate

Install the required packages:

pip install -r requirements.txt

6. Running the Application

Start the Streamlit application from the project root:

streamlit run app.py

Open the local Streamlit URL displayed in the terminal, typically:

http://localhost:8501

The chatbot interface can then be used to submit customer questions and evaluate the generated responses.

7. System Workflow

The application follows a modular processing pipeline:

A customer submits a message through the Streamlit interface.

The runtime validates and processes the message.

Relevant customer-service modules analyze the message.

Intent, sentiment, priority, and routing information are determined where applicable.

The knowledge-base retrieval component searches the available customer-service information.

The relevant information is returned to the response layer.

The chatbot presents the response to the customer.

This architecture keeps the individual responsibilities separated while allowing them to operate as one customer-service system.

8. Knowledge Base

The knowledge-base component is responsible for managing the information used by the chatbot for customer responses.

The project supports knowledge-base versioning so that updates can be managed without directly modifying the existing task modules.

Generated knowledge-base information is maintained under:

runtime_data/

9. Troubleshooting

Dependencies are missing

Activate the project virtual environment and run:

pip install -r requirements.txt

Dataset is not being used

Verify that the dataset exists at:

dataset/dataset.csv

Then restart the application and check the terminal output for runtime or ingestion errors.

Streamlit does not start

Make sure the command is being executed from the project root and that the virtual environment is active.

Chatbot does not return the expected knowledge

Verify that the knowledge-base process has completed successfully and that the latest dataset information is available to the active knowledge base.

10. Technology Stack

Component

Technology

Language

Python

Frontend

Streamlit

RAG Framework

LangChain

Vector Search

FAISS

Embeddings

Sentence Transformers

Knowledge Source

CSV

Development Environment

VS Code

11. Future Integration

The application is structured so that the chatbot runtime can be connected to other interfaces in the future, including a website or WordPress-based chat interface.

A future deployment can follow this architecture:

Website / WordPress
        │
        ▼
     Chat UI
        │
        ▼
Customer Service Runtime
        │
        ▼
Task Modules + RAG Knowledge Base

This allows the existing customer-service logic to be reused while changing the frontend or deployment environment.

12. Conclusion

The Customer Service AI Chatbot provides a modular framework for handling customer interactions through a combination of task-based processing and knowledge-base retrieval. Its separation between the user interface, runtime, task modules, and knowledge base makes the system easier to maintain and extend for future web or WordPress integration.
