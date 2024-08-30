import os
from dotenv import load_dotenv
import requests

load_dotenv()

api_key = os.getenv('OPENAI_API_KEY')

def generate_knowledge(text, file_name):
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    data = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant named Brum, generating knowledge about AI, Machine Learning, and Deep Learning in a structured format."},
            {"role": "user", "content": text}
        ]
    }
    response = requests.post(url, headers=headers, json=data)
    response_data = response.json()
    response_results = response_data['choices'][0]['message']['content'].strip()

    # Save the results to a .txt file
    with open(file_name, 'a') as file:
        file.write(response_results + '\n\n')
    
    return response_results

if __name__ == '__main__':
    while True:
        t = input('Enter topic (or type "exit" to quit): ')
        if t == 'exit':
            break
        result = generate_knowledge(t, 'ai_knowledge.txt')
        print(f'Assistant: {result}')
        print("Knowledge has been saved to ai_knowledge.txt.")
