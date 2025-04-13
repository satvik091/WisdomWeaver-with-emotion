import streamlit as st
import pandas as pd
import os
import json
from PIL import Image
from typing import Dict
import google.generativeai as genai
from transformers import pipeline

# ------------------------- GitaGeminiBot -------------------------

class GitaGeminiBot:
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash')  # More capable
        self.verses_db = self.load_gita_database()

    def load_gita_database(self) -> Dict:
        path = "bhagavad_gita_verses.csv"
        verses_df = pd.read_csv(path)
        verses_db = {}
        for _, row in verses_df.iterrows():
            chapter = f"chapter_{row['chapter_number']}"
            if chapter not in verses_db:
                verses_db[chapter] = {
                    "title": row['chapter_title'],
                    "verses": {}
                }
            verse_num = str(row['chapter_verse'])
            verses_db[chapter]["verses"][verse_num] = {
                "translation": row['translation']
            }
        return verses_db

    def format_response(self, raw_text: str) -> Dict:
        response = {
            "verse_reference": "",
            "sanskrit": "",
            "translation": "",
            "explanation": "",
            "application": ""
        }

        try:
            lines = raw_text.strip().split('\n')
            current = None
            for line in lines:
                if "Chapter" in line and "Verse" in line:
                    response["verse_reference"] = line
                elif line.startswith("Sanskrit:"):
                    response["sanskrit"] = line.replace("Sanskrit:", "").strip()
                elif line.startswith("Translation:"):
                    response["translation"] = line.replace("Translation:", "").strip()
                elif line.startswith("Explanation:"):
                    current = "explanation"
                    response[current] = line.replace("Explanation:", "").strip()
                elif line.startswith("Application:"):
                    current = "application"
                    response[current] = line.replace("Application:", "").strip()
                elif current:
                    response[current] += " " + line.strip()
        except Exception as e:
            response["verse_reference"] = "Error"
            response["translation"] = f"Error: {e}"
            response["explanation"] = "Please try again."
        return response

    def get_response(self, emotion: str, question: str) -> Dict:
        prompt = f"""
You are a spiritual teacher of the Bhagavad Gita.

User Emotion: **{emotion}**
User Question: **{question}**

Give a response rooted in the Bhagavad Gita. Follow this exact structure:
- Chapter X, Verse Y
- Sanskrit: ...
- Translation: ...
- Explanation: ...
- Application: (Simple words. Help the user connect the Gita's message to their feelings.)

⚠️ Only use real Bhagavad Gita verses with accurate chapter/verse numbers. Do not generate fake ones.
"""
        try:
            response = self.model.generate_content(prompt)
            return self.format_response(response.text or "")
        except Exception as e:
            return {
                "verse_reference": "Error",
                "translation": f"Gemini Error: {e}",
                "explanation": "Please try again.",
                "application": ""
            }

# ------------------------- Emotion Detection -------------------------

pipe = pipeline("image-classification", model="prithivMLmods/Facial-Emotion-Detection-SigLIP2")

def detect_emotion_from_image(image: Image.Image) -> str:
    try:
        results = pipe(image)
        return results[0]["label"] if results else "neutral"
    except:
        return "neutral"

# ------------------------- Streamlit App -------------------------

def initialize_state():
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'bot' not in st.session_state:
        st.session_state.bot = GitaGeminiBot(api_key=st.secrets["GEMINI_API_KEY"])

def main():
    st.set_page_config(page_title="Gita Wisdom Weaver", page_icon="🕉️", layout="wide")
    initialize_state()

    col1, col2 = st.columns([2, 1])

    with col1:
        st.title("🕉️ Gita Wisdom Based on Your Emotion")
        st.markdown("Upload your photo to sense emotion, ask your question, and receive divine guidance.")

        question = st.text_input("Ask your question")
        uploaded_image = st.file_uploader("Upload your face image (JPG/PNG)", type=["jpg", "png"])

        if st.button("✨ Ask with Emotion"):
            if not uploaded_image:
                st.warning("Please upload a photo to detect your emotion.")
            elif not question:
                st.warning("Please enter a question.")
            else:
                image = Image.open(uploaded_image)
                with st.spinner("Detecting emotion..."):
                    emotion = detect_emotion_from_image(image)

                with st.spinner(f"Reflecting with your emotion: {emotion}..."):
                    response = st.session_state.bot.get_response(emotion, question)
                    st.session_state.messages.append({"role": "user", "content": f"{question} (Feeling: {emotion})"})
                    st.session_state.messages.append({"role": "assistant", **response})
                    st.experimental_rerun()

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                if message["role"] == "user":
                    st.markdown(message["content"])
                else:
                    st.markdown(f"**{message['verse_reference']}**")
                    st.markdown(f"*{message['sanskrit']}*")
                    st.markdown(f"**Translation:** {message['translation']}")
                    st.markdown(f"**Explanation:** {message['explanation']}")
                    st.markdown(f"**Application:** {message['application']}")

    with col2:
        st.sidebar.title("📚 Browse Gita Chapters")
        chapters = list(st.session_state.bot.verses_db.keys())
        selected = st.sidebar.selectbox("Chapter", chapters, format_func=lambda c: f"{c}: {st.session_state.bot.verses_db[c]['title']}")
        for vnum, vdata in st.session_state.bot.verses_db[selected]["verses"].items():
            with st.sidebar.expander(f"Verse {vnum}"):
                st.markdown(vdata["translation"])

    st.markdown("---")
    st.markdown("""
🧘‍♂️ Powered by Emotion AI and the timeless wisdom of the Bhagavad Gita.
""")
