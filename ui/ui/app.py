import streamlit as st
import torch
import sys
import os

# Add ai_engine path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from ai_engine.model import StoryGenerator

# 🌟 Page Configuration
st.set_page_config(page_title="AI Story Creator", page_icon="📖", layout="centered")

# 🌈 Custom CSS Styling (Light Background, Dark Text)
st.markdown("""
    <style>
        .main {
            background-color: #e0f7fa;
            color: #000;
        }
        h1 {
            font-family: 'Comic Sans MS', cursive, sans-serif;
            color: #f57c00;
        }
        .story-box {
            background-color: #ffffff;
            padding: 20px;
            border-radius: 15px;
            border: 1px solid #ccc;
            font-size: 18px;
            line-height: 1.6;
            font-family: 'Georgia', serif;
            color: #000;
        }
        .caption {
            font-size: 14px;
            color: #555;
            text-align: center;
        }
    </style>
""", unsafe_allow_html=True)

# 📚 Title and Description
st.markdown("<h1 style='text-align: center;'>📖 AI-Powered Story Creator for Kids</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 18px;'>Let your imagination, Generate stories with a single click.</p>", unsafe_allow_html=True)

# ✏ Prompt Input
prompt = st.text_input("🌟 Start your story with...", value="Once upon a time in ")

# Instantiate model only once
generator = StoryGenerator()

# ✨ Generate Button
if st.button("✨ Generate Story"):
    with st.spinner("Generating a magical story... 🪄"):
        full_story = generator.generate(
            prompt=prompt,
            max_new_tokens=300,  # ⬅ Short and readable
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.2
        )
    st.subheader("📝 Your Generated Story")
    st.markdown(f"<div class='story-box'>{full_story}</div>", unsafe_allow_html=True)
