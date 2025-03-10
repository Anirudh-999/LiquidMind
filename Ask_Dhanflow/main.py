import gradio as gr
import pandas as pd
import os
import json
from cryptography.fernet import Fernet
from langchain.schema.messages import HumanMessage
import google.generativeai as genai
import psycopg2
import sqlalchemy

KEY_FILE = "secret.key"

DB_CONFIG = {
    "host": "liquidmind.postgres.database.azure.com",
    "port": 5432,
    "database": "liquidminddb",
    "user": "lm_admin",
    "password": "Place password here",
}

msme_id = "msme id here"

try:
    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()

    check_msme_query = "SELECT EXISTS(SELECT 1 FROM msme WHERE msme_id = %s)"
    cursor.execute(check_msme_query, (msme_id,))
    msme_exists = cursor.fetchone()[0]

    if not msme_exists:
        raise ValueError(f"MSME ID {msme_id} not found in database")
    
    query1 = "SELECT * FROM invoice WHERE msme_id = %s"

    query2 = "SELECT premium_days FROM msme WHERE msme_id = %s"
    cursor.execute(query1, (msme_id,)) 
    cursor.execute(query2, (msme_id,)) 

    premium_days = cursor.fetchone()[0]
    print(premium_days)

    columns = [desc[0] for desc in cursor.description]
    rows = cursor.fetchall()

    df = pd.DataFrame(rows, columns=columns)
    csv_directory = os.path.join(os.path.dirname(__file__), 'data', 'csv_files')
    os.makedirs(csv_directory, exist_ok=True)
    output_file = os.path.join(csv_directory, f"invoice_msme_{msme_id}.csv")
    df.to_csv(output_file, index=False)
    print(f"Data saved to {output_file}")

    cursor.close()
    conn.close()

except Exception as e:
    print("Error:", e)

CSV_DATA_PATH = output_file

def cleanup_session():
    try:
        if os.path.exists(CSV_DATA_PATH):
            os.remove(CSV_DATA_PATH)
    except Exception as e:
        print(f"Error cleaning up CSV file: {e}")

class ChatGemini:
    def __init__(self, model_name: str, credentials_path: str, generation_config: dict):
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path
        genai.configure(api_key="Replace with your API key")  
        self.model = genai.GenerativeModel(
            model_name=model_name, 
            generation_config=generation_config
        )

    def invoke(self, input_data) -> dict:
        try:
        # Parse input
            payload = json.loads(input_data[0].content)
            encrypted_csv = payload.get("encrypted_csv")
            password = payload.get("password")
            user_task = payload.get("topic")

            if not encrypted_csv or not password or not user_task:
                raise ValueError("Missing required data")
        # Decrypt CSV data
            cipher = Fernet(password.encode())
            decrypted_csv_data = cipher.decrypt(encrypted_csv.encode())
        # Load CSV into Pandas
            csv_data_path = "temp_decrypted.csv"
            with open(csv_data_path, "wb") as file:
                file.write(decrypted_csv_data)

            data = pd.read_csv(csv_data_path)
            os.remove(csv_data_path)
        # Prepare prompt
            csv_data_str = data.to_string(index=False)
            prompt = f'''
            The response should not exceed 70 words and it should be in bullet points and no special formatting 
            Task: {user_task}\n\nRelevant CSV Data: \n{csv_data_str}
            The response should be such a way that it should be understood by a common man'''
        # Send prompt to LLM
            chat_session = self.start_chat_session()
            response = chat_session.send_message(prompt)
        # Re-encrypt processed response
            encrypted_response = cipher.encrypt(response.text.encode())

            return {
            "response": encrypted_response.decode(),  
            "status": "success",
            }

        except Exception as e:
            return {"response": str(e), "status": "failed"}


    def start_chat_session(self):
        return self.model.start_chat(history=[])

# Key management
def generate_key():
    if not os.path.exists(KEY_FILE):
        key = Fernet.generate_key()
        with open(KEY_FILE, "wb") as key_file:
            key_file.write(key)

def load_key():
    with open(KEY_FILE, "rb") as key_file:
        return key_file.read()

generate_key()
key = load_key()
cipher = Fernet(key)

# Encrypt the CSV file
def encrypt_csv(file_path):
    with open(file_path, "rb") as file:
        encrypted_data = cipher.encrypt(file.read())
    return encrypted_data

encrypted_csv = encrypt_csv(CSV_DATA_PATH)

# Initialize Gemini LLM
generation_config = {
    "temperature": 0.4,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}
gemini_model = ChatGemini(
    model_name="gemini-1.5-flash",
    credentials_path="place your credentials path",
    generation_config=generation_config,
)

# Gradio function
def process_input(topic):
    try:
        # Validate input
        if not topic.strip():
            return "Topic cannot be empty."

        # Prepare payload
        payload = {
            "encrypted_csv": encrypted_csv.decode(),
            "password": key.decode(),
            "topic": topic.strip(),
        }

        # Call Gemini LLM
        payload_json = json.dumps(payload)
        result = gemini_model.invoke([HumanMessage(content=payload_json)])

        # Decrypt response
        encrypted_response = result.get("response", "").encode()
        decrypted_response = cipher.decrypt(encrypted_response).decode()

        # Format the response
        return decrypted_response
    except Exception as e:
        return f"An error occurred: {str(e)}"

# Gradio interface
interface = gr.Interface(
    fn=process_input,
    inputs=gr.Textbox(label="Enter Topic"),
    outputs=gr.Textbox(label="Gemini Response"),
    title="Gemini Gradio App",
    description="Enter a topic to interact with the Gemini LLM and see the processed response.",
)

if __name__ == "__main__":
    interface.launch(share=True)
