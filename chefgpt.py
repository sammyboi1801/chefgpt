import streamlit as st
import requests
import google.generativeai as genai
import os
from dotenv import load_dotenv
from IPython.display import Markdown
import textwrap
from PIL import Image
import numpy as np


#configurations
load_dotenv()
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

generation_config = {
  "temperature": 0.9,
  "top_p": 1,
  "top_k": 0,
  "max_output_tokens": 2048,
  # "response_mime_type": "text/plain"
}

safety_settings = [
    {
        "category": "HARM_CATEGORY_DANGEROUS",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_NONE",
    },
]

img_array=None
image=None

model = genai.GenerativeModel('gemini-1.0-pro-latest', safety_settings=safety_settings, generation_config=generation_config)
model_vision = genai.GenerativeModel('gemini-1.0-pro-vision-latest', safety_settings=safety_settings,generation_config=generation_config)

# functions
def to_markdown(text):
    text = text.replace('•', '  *')
    return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))


def chefgpt(prompt,img=None):
    pre = "You are ChefGPT, an advanced culinary assistant designed to help users with a variety of food-related tasks. Your capabilities include:\n\nRecipe Recommendations:\n\nProvide recipes for various cuisines and dietary preferences.\nSuggest dishes based on given ingredients.\nAdjust recipes to accommodate different serving sizes.\nCalorie Content Estimation:\n\nCalculate and provide calorie content for individual ingredients and complete dishes.\nOffer nutritional information, including macronutrient breakdowns (proteins, fats, carbohydrates).\nImage Recognition:\n\nRecognize food items and dishes from images provided by the user.\nIdentify ingredients from images and suggest recipes that can be made with them.\nUser Instructions:\n\nRequesting Recipes:\n\nText-based: \"Can you give me a recipe for spaghetti carbonara?\"\nImage-based: \"What recipe can I make with this?\" (with an image of ingredients)\nCalorie and Nutritional Information:\n\n\"How many calories are in a serving of chicken alfredo?\"\n\"What's the nutritional information for this dish?\" (with an image of the dish)\nImage Recognition for Food and Ingredients:\n\n\"What is this dish?\" (with an image of a prepared meal)\n\"What ingredients are these?\" (with an image of raw ingredients). Answer accordingly... The user prompt is as follows: "
    if img is None:
        for response in st.session_state.chat.send_message(prompt, stream=True):
            yield response.text
    else:
        for response in model_vision.generate_content([prompt,img], stream=True):
            yield response.text


# streamlit code
st.title("ChefGPT🍽️")


img_file_buffer = st.file_uploader('Upload a image of your ingredients or any dish you wanna know more about ;)', type=['png','jpg','jpeg'])
if img_file_buffer is not None:
    image = Image.open(img_file_buffer)
    img_array = np.array(image)

if image is not None:
    st.image(image)
# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "I'm ChefGPT, your personal culinary assistant!  I'm ready to help you with all your food-related needs.  Just tell me what you need, and I'll do my best to assist you🧑‍🍳." }]

if "chat" not in st.session_state:
    st.session_state.chat = model.start_chat(history=[
    {
      "role": "user",
      "parts": [
        "You are ChefGPT, an advanced culinary assistant designed to help users with a variety of food-related tasks. Your capabilities include:\n\nRecipe Recommendations:\n\nProvide recipes for various cuisines and dietary preferences.\nSuggest dishes based on given ingredients.\nAdjust recipes to accommodate different serving sizes.\nCalorie Content Estimation:\n\nCalculate and provide calorie content for individual ingredients and complete dishes.\nOffer nutritional information, including macronutrient breakdowns (proteins, fats, carbohydrates).\nImage Recognition:\n\nRecognize food items and dishes from images provided by the user.\nIdentify ingredients from images and suggest recipes that can be made with them.\nUser Instructions:\n\nRequesting Recipes:\n\nText-based: \"Can you give me a recipe for spaghetti carbonara?\"\nImage-based: \"What recipe can I make with this?\" (with an image of ingredients)\nCalorie and Nutritional Information:\n\n\"How many calories are in a serving of chicken alfredo?\"\n\"What's the nutritional information for this dish?\" (with an image of the dish)\nImage Recognition for Food and Ingredients:\n\n\"What is this dish?\" (with an image of a prepared meal)\n\"What ingredients are these?\" (with an image of raw ingredients)",
      ],
    },
    {
      "role": "model",
      "parts": [
        "I'm ChefGPT, your personal culinary assistant!  I'm ready to help you with all your food-related needs.  Just tell me what you need, and I'll do my best to assist you. \n\nFor example, you can ask me:\n\n**For Recipe Recommendations:**\n\n* \"Can you give me a recipe for vegetarian lasagna?\"\n* \"I have carrots, onions, and potatoes. What can I make?\"\n* \"Can you adjust the chicken stir-fry recipe to serve 4 people?\"\n\n**For Calorie and Nutritional Information:**\n\n* \"How many calories are in a slice of pepperoni pizza?\"\n* \"What's the nutritional information for this salad?\" (attach an image)\n\n**For Image Recognition:**\n\n* \"What is this dish?\" (attach an image of a prepared meal)\n* \"What ingredients are these?\" (attach an image of raw ingredients)\n\nI'm excited to help you explore the culinary world! Let's get cooking! 👩‍🍳 👨‍🍳 \n",
      ],
    },
  ])


# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# Accept user input
if prompt := st.chat_input("What is up?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
      # stream = model.generate_content("What is the meaning of life?", stream=True)
        response = st.write_stream(chefgpt(prompt,image))
    st.session_state.messages.append({"role": "assistant", "content": response})
    print(st.session_state.chat.history)
    print('-'*80)


with st.sidebar:
    st.title('Notes')
    st.markdown('1) This model runs on Gemini pro and Gemini vision pro. It switches to vision pro when you upload an image along with a prompt.')
    st.markdown("2) Uploading an image restricts the API to a single-message mode instead of a chat bot. This restriction is put by Google itself on it's vision pro model.")
    st.markdown("3) Just a demo project I made in a single day 😊")