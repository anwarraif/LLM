import psycopg2
from psycopg2 import sql

# Database credentials
DB_NAME = "rag_open_source"
DB_USER = "your_user"
DB_PASSWORD = "your_password"
DB_HOST = "localhost"
DB_PORT = "8012"

# Function to create the database and new_dbname 
def setup_database_and_tables():
    # Step 1: Connect to PostgreSQL to create the database if it doesn't exist
    try:
        conn = psycopg2.connect(dbname="postgres", user=DB_USER, password=DB_PASSWORD, host=DB_HOST, port=DB_PORT)
        conn.autocommit = True  # Required for database creation
        cursor = conn.cursor()

        # Check if the database already exists
        cursor.execute("SELECT 1 FROM pg_database WHERE datname = %s", (DB_NAME,))
        exists = cursor.fetchone()

        # Create the database if it doesn't exist
        if not exists:
            cursor.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(DB_NAME)))
            print(f"Database '{DB_NAME}' created successfully.")
        else:
            print(f"Database '{DB_NAME}' already exists.")
        
        cursor.close()
        conn.close()
    except Exception as e:
        print(f"Error creating database: {e}")
        return

    # Step 2: Connect to the created database and create the tables if they don't exist
    try:
        conn = psycopg2.connect(dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD, host=DB_HOST, port=DB_PORT)
        cursor = conn.cursor()

        # Enable pgvector extension
        cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")

        # Create chat_history table with instruction, input_data, response, and vectors
        create_chat_history_table = """
        CREATE TABLE IF NOT EXISTS chat_history (
            id SERIAL PRIMARY KEY,
            instruction TEXT NOT NULL,
            input_data TEXT NOT NULL,
            response TEXT NOT NULL,
            input_vector VECTOR(768),  -- Adjusted dimension to 1024
            response_vector VECTOR(768),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
        cursor.execute(create_chat_history_table)

        # Create cached_chat table with only chat_history_id
        create_cached_chat_table = """
        CREATE TABLE IF NOT EXISTS cached_chat (
            id SERIAL PRIMARY KEY,
            chat_history_id INT REFERENCES chat_history(id) ON DELETE CASCADE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
        cursor.execute(create_cached_chat_table)

        # Create analytics table
        create_analytics_table = """
        CREATE TABLE IF NOT EXISTS analytics (
            id SERIAL PRIMARY KEY,
            type VARCHAR(50) NOT NULL,  -- 'like', 'dislike', 'regenerate'
            chat_history_id INT REFERENCES chat_history(id) ON DELETE CASCADE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
        cursor.execute(create_analytics_table)

        # Create knowledge table
        create_knowledge_table = """
        CREATE TABLE IF NOT EXISTS knowledge (
            id SERIAL PRIMARY KEY,
            document_id TEXT UNIQUE NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
        cursor.execute(create_knowledge_table)

        # Create knowledge_vector table
        create_knowledge_vector_table = """
        CREATE TABLE IF NOT EXISTS knowledge_vector (
            id SERIAL PRIMARY KEY,
            knowledge_id INT REFERENCES knowledge(id) ON DELETE CASCADE,
            chunk_text TEXT NOT NULL,
            chunk_vector VECTOR(768) NOT NULL,  -- Adjusted dimension to 1024
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
        cursor.execute(create_knowledge_vector_table)

        conn.commit()
        print("All tables created or already exist.")

        cursor.close()
        conn.close()
    except Exception as e:
        print(f"Error creating tables: {e}")

# Call the function to setup the database and tables
setup_database_and_tables() 