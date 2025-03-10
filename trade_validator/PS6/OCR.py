from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.credentials import AzureKeyCredential
import re

# Mock credentials for Azure DocumentAnalysisClient (replace with actual values)
AZURE_ENDPOINT = "https://liquidmindinvoice.cognitiveservices.azure.com/"
AZURE_KEY = "137454561e0c45598dee07adc71e75d1"

def extract_details(file_path):
    try:
        # Create a DocumentAnalysisClient instance
        client = DocumentAnalysisClient(endpoint=AZURE_ENDPOINT, credential=AzureKeyCredential(AZURE_KEY))

        # Open and process the document
        with open(file_path, "rb") as f:
            poller = client.begin_analyze_document("prebuilt-read", document=f)
            result = poller.result()

        extracted_data = []

        for page in result.pages:
            for line in page.lines:
                text = line.content.strip()
                extracted_data.append(text)

        return "\n".join(extracted_data)

    except Exception as e:
        error_message = str(e)
        if "InvalidContent" in error_message:
            return {"Error": "The file is corrupted or not in a supported format. Please upload a valid file."}
        return {"Error": f"An unexpected error occurred: {error_message}"}
