import psycopg2
import os
from psycopg2.extras import Json
from datetime import datetime

class VectorStore:
    def __init__(self):
        self.connection = psycopg2.connect(
            dbname="db_rag_assignment",
            user=os.getenv("PG_USER"),
            password=os.getenv("PG_PASSWORD"),
            host=os.getenv("PG_HOST"),
            port=os.getenv("PG_PORT")
        )
        self.cursor = self.connection.cursor()

    def create_tables(self):
        self.cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS knowledge (
                id SERIAL PRIMARY KEY,
                embedding VECTOR(1536),
                content TEXT
            );

            CREATE TABLE IF NOT EXISTS chat_sessions (
                session_id SERIAL PRIMARY KEY,
                history JSONB
            );

            CREATE TABLE IF NOT EXISTS chat_history (
                id SERIAL PRIMARY KEY,
                chat_session_id INT REFERENCES chat_sessions(session_id),
                chat TEXT,
                ai_answer TEXT,
                chat_embedding VECTOR(1536),
                ai_answer_embedding VECTOR(1536)
            );

            CREATE TABLE IF NOT EXISTS token_counter (
                id SERIAL PRIMARY KEY,
                token_type VARCHAR(50),
                tokens_used INT,
                timestamp TIMESTAMP
            );
        """)
        self.connection.commit()

    def store_embedding(self, embedding, content, tokens_used):
        embedding = str(embedding)
        self.cursor.execute("""
            INSERT INTO knowledge (embedding, content)
            VALUES (%s, %s)
        """, (embedding, content))
        self.connection.commit()
        # self.store_token_count('embedding_input', tokens_used)

    def query_similar(self, embedding, limit=1):
        embedding = str(embedding)
        self.cursor.execute("""
            SELECT content FROM knowledge
            ORDER BY embedding <-> %s
            LIMIT %s
        """, (embedding, limit))
        return self.cursor.fetchall()

    def store_session(self, history):
        self.cursor.execute("""
            INSERT INTO chat_sessions (history)
            VALUES (%s) RETURNING session_id
        """, [Json(history)])
        session_id = self.cursor.fetchone()[0]
        self.connection.commit()
        return session_id

    def get_session(self, session_id):
        self.cursor.execute("""
            SELECT history FROM chat_sessions WHERE session_id = %s
        """, (session_id,))
        return self.cursor.fetchone()[0]

    def update_session(self, session_id, history):
        self.cursor.execute("""
            UPDATE chat_sessions SET history = %s WHERE session_id = %s
        """, [Json(history), session_id])
        self.connection.commit()

    def store_chat_history(self, session_id, chat, ai_answer, chat_embedding, ai_answer_embedding):
        self.cursor.execute("""
            INSERT INTO chat_history (chat_session_id, chat, ai_answer, chat_embedding, ai_answer_embedding)
            VALUES (%s, %s, %s, %s, %s)
        """, (session_id, chat, ai_answer, chat_embedding, ai_answer_embedding))
        self.connection.commit()

    def query_chat_history(self, session_id, embedding, limit=1):
        embedding=str(embedding)
        self.cursor.execute("""
            SELECT chat, ai_answer FROM chat_history
            WHERE chat_session_id = %s
            ORDER BY chat_embedding <-> %s
            LIMIT %s
        """, (session_id, embedding, limit))
        return self.cursor.fetchall()

    def store_token_count(self, token_type, tokens_used):
        self.cursor.execute("""
            INSERT INTO token_counter (token_type, tokens_used, timestamp)
            VALUES (%s, %s, %s)
        """, (token_type, tokens_used, datetime.now()))
        self.connection.commit()

    def query_token_usage(self, token_type, start_date, end_date):
        self.cursor.execute("""
            SELECT SUM(tokens_used) FROM token_counter
            WHERE token_type = %s AND timestamp BETWEEN %s AND %s
        """, (token_type, start_date, end_date))
        return self.cursor.fetchone()[0] or 0