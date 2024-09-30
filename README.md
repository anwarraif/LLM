# Document Retrieval and Embedding System using RAG Framework - Llama Model Fine Tune

## Chat Completions - Retrieval Augmented Generation (RAG) - Llama Model Fine Tune

This project aims to develop an advanced Retrieval-Augmented Generation (RAG) system designed to intelligently extract text from documents, transform the text into high-dimensional embeddings using OpenAI's API, and store these embeddings in a PostgreSQL database + DBeaver. The system facilitates efficient retrieval of relevant information by comparing new queries with stored embeddings, enabling precise and contextually accurate responses. This solution is particularly useful for applications such as document indexing, knowledge management, and automated question-answering systems, where quick and relevant information retrieval is crucial. The Project focuses on storing and retrieving knowledge specifically about Cirebon cuisine. You can generate any knowledge from generate_knowledge.py actually using open AI API GPT.

The system extracts text related to Cirebon cuisine and converts this information into high-dimensional embeddings using a custom Llama model for text embedding, which has been fine-tuned for the Indonesian language. These embeddings are then stored in a PostgreSQL database enhanced with pgvector for efficient vector storage and querying. The Llama model's extended context window allows for better comprehension of longer and more complex input, making it particularly suited for retrieving relevant information in context.

The entire application is deployed using Docker and Docker Compose, ensuring seamless setup and management of the environment. It is powered by FastAPI and served via Uvicorn, providing a fast, scalable interface for querying and retrieving relevant culinary information. This setup allows users to explore and preserve regional culinary knowledge, such as Cirebon cuisine, through AI-driven techniques.

The system's overall flow follows a Retrieval-Augmented Generation (RAG) process, where text is first extracted from a document, converted into embeddings using the Llama text embedding model, and stored in the vector database. Later, based on user queries, these embeddings are used to find similar or relevant content. This method is particularly effective for tasks like question answering, where the system retrieves pertinent information from the knowledge base and generates an informative response based on the input query. The combination of retrieval and generation ensures that the system can provide accurate and contextually relevant answers.

The evaluation system addapt with reaction_app.py that can be used for adding like, dislike, or regenerate button for online evaluation metrics. Hope this project can be improve and collaborate with the readers in the feature.

## Requirements Version
- fastapi==0.112.2
- psycopg2==2.9.9
- psycopg2-binary==2.9.9
- requests==2.32.3
- python-dotenv==1.0.1
- uvicorn==0.30.6
- pypdf2==3.0.1
- python-multipart==0.0.9

## Usage
First, you need to set up the pgvector extension for images in Docker. I'm using Docker Desktop (link provided below). Then, run Docker Compose to connect to DBeaver. After that, test the PostgreSQL connection in DBeaver using the port, user, and password specified in the docker-compose.yaml file. Once that's done, run the code in initial_db.py to create the Vector Database in PostgreSQL. Next, run main.py and open the port for FastAPI.

- docker image : https://www.docker.com/blog/how-to-use-the-postgres-docker-official-image/
- Setting Llama : https://github.com/abetlen/llama-cpp-python & https://stackoverflow.com/questions/77267346/error-while-installing-python-package-llama-cpp-python

#Llama
#Fine-tunning
#RAG
#GPT
#FastAPI
#Retrieval Information
