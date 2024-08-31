# Document Retrieval and Embedding System using RAG Framework
## Chat Completions - Retrieval Augmented Generation (RAG)

This project aims to develop an advanced Retrieval-Augmented Generation (RAG) system designed to intelligently extract text from documents, transform the text into high-dimensional embeddings using OpenAI's API, and store these embeddings in a PostgreSQL database + DBeaver. The system facilitates efficient retrieval of relevant information by comparing new queries with stored embeddings, enabling precise and contextually accurate responses. This solution is particularly useful for applications such as document indexing, knowledge management, and automated question-answering systems, where quick and relevant information retrieval is crucial. The Project focuses on storing and retrieving knowledge specifically about Cirebon cuisine. You can generate any knowledge from `generate_knowledge.py`.

The system extracts text related to Cirebon cuisine, converts this information into high-dimensional embeddings using OpenAI's API, and stores these embeddings in a PostgreSQL database enhanced with pgvector for efficient vector storage and querying. The deployment is managed using Docker and Docker Compose, ensuring a seamless setup of the environment. The entire application is powered by FastAPI and served using Uvicorn, offering a robust, scalable, and fast interface for querying and retrieving relevant culinary information. This project is ideal for those interested in preserving and exploring regional culinary knowledge through modern AI-driven techniques.

The overall flow represents a Retrieval-Augmented Generation (RAG) process where text is extracted from a document, converted into embeddings, stored in a vector database, and later queried to find similar content based on embeddings. This method is useful in tasks like question answering, where the system retrieves relevant information from a database of knowledge based on the input query.

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
First, you need to set up the pgvector extension for images in Docker. I'm using Docker Desktop (link provided below). Then, run Docker Compose to connect to DBeaver. After that, test the PostgreSQL connection in DBeaver using the port, user, and password specified in the docker-compose.yaml file. Once that's done, run the code in initial_db.py to create the Vector Database in PostgreSQL. Next, run main.py and open the port for FastAPI. You can refer to the tutorial video on the drive for the testing process.

docker image : https://www.docker.com/blog/how-to-use-the-postgres-docker-official-image/
link tutorial video : https://drive.google.com/drive/folders/1PuaNL4N2WXpIz9UfygSpemTc4R4F5LJ3?usp=sharing

#RAG
#GPT
#FastAPI
#Retrieval Information