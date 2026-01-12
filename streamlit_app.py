import streamlit as st
import requests
import base64
from PIL import Image
import io
import json
import os


# -------------------------------
# Config
# -------------------------------
API_URL = "https://router.huggingface.co/v1/chat/completions"
MODEL_NAME = "Qwen/Qwen3-VL-8B-Instruct:novita"

HF_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN") if "HUGGINGFACE_API_TOKEN" in os.environ else st.secrets["HUGGINGFACE_API_TOKEN"]

headers = {
    "Authorization": f"Bearer {HF_TOKEN}",
    "Content-Type": "application/json",
}

# -------------------------------
# Utils
# -------------------------------
def image_to_data_url(image: Image.Image) -> str:
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    image_bytes = buffered.getvalue()
    image_base64 = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:image/jpeg;base64,{image_base64}"

def stream_chat_completion(payload):
    response = requests.post(
        API_URL,
        headers=headers,
        json=payload,
        stream=True,
        timeout=60,
    )

    for line in response.iter_lines():
        if line:
            decoded = line.decode("utf-8")
            if decoded.startswith("data: "):
                data = decoded.replace("data: ", "")
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                    delta = chunk["choices"][0]["delta"]
                    if "content" in delta:
                        yield delta["content"]
                except Exception:
                    pass

# -------------------------------
# UI
# -------------------------------
st.set_page_config(page_title="Vision LLM Demo", page_icon="🤖")
st.title("🤖 Vision LLM – Hugging Face")
info_client = st.context.cookies
st.write("Info de l'utilisateur :", info_client)
question = st.text_input("Pose ta question", placeholder="What is this equipment and what is it used for?")
uploaded_file = st.file_uploader("Upload une image", type=["jpg", "jpeg", "png"], accept_multiple_files=False)

image = None
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Image uploadée", use_column_width=True)

if st.button("Analyser") and question:
    with st.spinner("Analyse en cours..."):
        content_list = [{"type": "text", "text": question}]

        if image:
            image_data_url = image_to_data_url(image)
            content_list.append({"type": "image_url", "image_url": {"url": image_data_url}})

        payload = {
            "model": MODEL_NAME,
            "messages": [{"role": "user", "content": content_list}],
            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
            "stream": True,
        }

        response_container = st.empty()
        full_response = ""

        for token in stream_chat_completion(payload):
            full_response += token
            response_container.markdown(full_response)
