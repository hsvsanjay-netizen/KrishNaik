import streamlit as st
from PIL import Image
from gtts import gTTS
import tempfile
import os

from openai import OpenAI
from groq import Groq
from google.genai import Client as GeminiClient
from google.genai.errors import ClientError


def format_gemini_error(error: ClientError) -> str:
    if getattr(error, 'message', None):
        message = error.message
    else:
        message = str(error)

    if getattr(error, 'status', None):
        status = error.status
    else:
        status = 'Unknown'

    if isinstance(message, str) and 'expired' in message.lower():
        return "Gemini API key expired. Please renew your Gemini API key."
    if isinstance(message, str) and 'quota' in message.lower():
        return "Gemini quota exceeded. Please check your Google Cloud billing and rate limits."
    if isinstance(message, str) and (
        'not found' in message.lower()
        or 'unsupported' in message.lower()
        or 'invalid' in message.lower()
    ):
        return (
            f"Gemini API error ({status}): {message}. "
            "Check GEMINI_MODEL in your environment or try a supported Gemini model such as "
            "gemini-2.5-flash or gemini-3.5-flash."
        )
    return f"Gemini API error ({status}): {message}"


def load_dotenv(dotenv_path=None):
    if dotenv_path is None:
        dotenv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if not os.path.exists(dotenv_path):
        return
    with open(dotenv_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value

load_dotenv()

# =====================================================
# API KEYS
# =====================================================

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OPENAI_MODEL_DEFAULT = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
GROQ_MODEL_DEFAULT = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
GEMINI_MODEL_DEFAULT = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

# =====================================================
# CLIENT HELPERS
# =====================================================

def get_openai_client():
    if not OPENAI_API_KEY:
        return None
    return OpenAI(api_key=OPENAI_API_KEY)


def get_groq_client():
    if not GROQ_API_KEY:
        return None
    return Groq(api_key=GROQ_API_KEY)


def get_gemini_client():
    if not GEMINI_API_KEY:
        return None
    return GeminiClient(api_key=GEMINI_API_KEY)

# =====================================================
# STREAMLIT UI
# =====================================================

st.set_page_config(
    page_title="Multimodal AI Web App",
    layout="wide"
)

st.title("🎯 Multimodal AI Web Application")

modality = st.sidebar.selectbox(
    "Choose Modality",
    [
        "Text to Text",
        "Image to Text",
        "Audio to Text",
        "Text to Audio"
    ]
)

model_choice = st.sidebar.selectbox(
    "Choose Model",
    [
        "OpenAI GPT",
        "Groq Llama3",
        "Gemini"
    ]
)

openai_model = st.sidebar.text_input(
    "OpenAI Model",
    value=OPENAI_MODEL_DEFAULT,
    help="Choose OpenAI model. gpt-3.5-turbo is the best free-tier default."
)

groq_model = st.sidebar.text_input(
    "Groq Model",
    value=GROQ_MODEL_DEFAULT,
    help="Choose Groq model. llama-3.1-8b-instant is the recommended free-tier model."
)

gemini_model = st.sidebar.text_input(
    "Gemini Model",
    value=GEMINI_MODEL_DEFAULT,
    help="Choose Gemini model. gemini-2.5-flash is the supported stable default in this client."
)

# =====================================================
# TEXT TO TEXT
# =====================================================

if modality == "Text to Text":

    user_input = st.text_area("Enter your prompt")

    if st.button("Generate Response"):

        output = None

        if model_choice == "OpenAI GPT":

            openai_client = get_openai_client()
            if openai_client is None:
                st.error("OPENAI_API_KEY is not set. Please set it in your environment before using OpenAI GPT.")
            else:
                response = openai_client.chat.completions.create(
                    model=openai_model or OPENAI_MODEL_DEFAULT,
                    messages=[
                        {
                            "role": "user",
                            "content": user_input
                        }
                    ]
                )

                output = response.choices[0].message.content

        elif model_choice == "Groq Llama3":

            groq_client = get_groq_client()
            if groq_client is None:
                st.error("GROQ_API_KEY is not set. Please set it in your environment before using Groq Llama3.")
            else:
                response = groq_client.chat.completions.create(
                    model=groq_model or GROQ_MODEL_DEFAULT,
                    messages=[
                        {
                            "role": "user",
                            "content": user_input
                        }
                    ]
                )

                output = response.choices[0].message.content

        elif model_choice == "Gemini":

            gemini_client = get_gemini_client()
            if gemini_client is None:
                st.error("GEMINI_API_KEY is not set. Please set it in your environment before using Gemini.")
            else:
                try:
                    response = gemini_client.models.generate_content(
                        model=gemini_model or GEMINI_MODEL,
                        contents=user_input
                    )
                    output = response.text
                except ClientError as e:
                    st.error(format_gemini_error(e))
                    output = None

        if output is not None:
            st.success("AI Response")
            st.write(output)

# =====================================================
# IMAGE TO TEXT
# =====================================================

elif modality == "Image to Text":

    uploaded_image = st.file_uploader(
        "Upload Image",
        type=["png", "jpg", "jpeg"]
    )

    if uploaded_image:

        image = Image.open(uploaded_image)

        st.image(image, caption="Uploaded Image")

        if st.button("Generate Description"):

            gemini_client = get_gemini_client()
            if gemini_client is None:
                st.error("GEMINI_API_KEY is not set. Please set it in your environment before using Gemini.")
            else:
                try:
                    response = gemini_client.models.generate_content(
                        model=gemini_model or GEMINI_MODEL,
                        contents=[
                            "Describe this image in detail",
                            image
                        ]
                    )
                    st.success("Image Description")
                    st.write(response.text)
                except ClientError as e:
                    st.error(format_gemini_error(e))

# =====================================================
# AUDIO TO TEXT
# =====================================================

elif modality == "Audio to Text":

    uploaded_audio = st.file_uploader(
        "Upload Audio",
        type=["mp3", "wav"]
    )

    if uploaded_audio:

        st.audio(uploaded_audio)

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".mp3"
        ) as tmp:

            tmp.write(uploaded_audio.read())

            temp_audio_path = tmp.name

        if st.button("Transcribe Audio"):

            openai_client = get_openai_client()
            if openai_client is None:
                st.error("OPENAI_API_KEY is not set. Please set it in your environment before using OpenAI audio transcription.")
            else:
                with open(temp_audio_path, "rb") as audio_file:
                    result = openai_client.audio.transcriptions.create(
                        model="whisper-1",
                        file=audio_file
                    )

                st.success("Transcription")
                st.write(result.text)

# =====================================================
# TEXT TO AUDIO
# =====================================================

elif modality == "Text to Audio":

    text = st.text_area("Enter text")

    if st.button("Generate Audio"):

        tts = gTTS(text=text, lang="en")

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".mp3"
        ) as tmp:

            tts.save(tmp.name)

            with open(tmp.name, "rb") as audio_file:

                st.audio(
                    audio_file.read(),
                    format="audio/mp3"
                )