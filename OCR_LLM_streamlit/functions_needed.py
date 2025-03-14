import google.generativeai as genai
def oppo(date,operation):

    genai.configure(api_key="add your api key")

    # Create the model
    generation_config = {
      "temperature": 1,
      "top_p": 0.95,
      "top_k": 40,
      "max_output_tokens": 8192,
      "response_mime_type": "text/plain",
    }

    model = genai.GenerativeModel(
      model_name="gemini-1.5-flash",
      generation_config=generation_config,
    )

    chat_session = model.start_chat(
      history=[
      ]
    )
    response = chat_session.send_message(f"give me the {operation} of charge in the following date : {date}")

    return response.text

def edit(textalign):

    genai.configure(api_key="add your api key")

    # Create the model
    generation_config = {
      "temperature": 1,
      "top_p": 0.95,
      "top_k": 40,
      "max_output_tokens": 8192,
      "response_mime_type": "text/plain",
    }

    model = genai.GenerativeModel(
      model_name="gemini-1.5-flash",
      generation_config=generation_config,
    )

    chat_session = model.start_chat(
      history=[
      ]
    )
    response = chat_session.send_message(f"give me this information in the form of dictionary dont add date again,1.item-price item is key: {textalign}")

    return response.text
