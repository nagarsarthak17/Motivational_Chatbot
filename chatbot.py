import random
import json
import pickle
import numpy as np
import tensorflow as tf
import nltk
from nltk.stem import WordNetLemmatizer
from keras.models import load_model
import langid
from googletrans import Translator


# Load necessary resources
lemmatizer = WordNetLemmatizer()
model = load_model('.venv/chatbot_motivational.h5')
translator = Translator()


# Load intents based on language
def load_intents(language):
   filename = f'intents_{language}.json'
   try:
       with open(filename, 'r', encoding='utf-8') as file:
           intents = json.load(file)
       return intents
   except FileNotFoundError:
       print(f"Intents file for {language} not found.")
       return None


# Preprocess input text
def preprocess_input(input_text):
   tokens = nltk.word_tokenize(input_text)
   lemmatized_tokens = [lemmatizer.lemmatize(word.lower()) for word in tokens]
   return lemmatized_tokens


# Get response from the model
def get_response(input_text, lang):
   input_text = input_text.lower()
   processed_input = preprocess_input(input_text)


   # Translate non-English input to English
   if lang != 'en':
       translated_text = translator.translate(input_text, src=lang, dest='en').text
       processed_input = preprocess_input(translated_text)


   # Generate model input
   bag = [0] * len(words)
   for w in processed_input:
       for i, word in enumerate(words):
           if word == w:
               bag[i] = 1


   # Get prediction from model
   results = model.predict(np.array([bag]))[0]
   results_index = np.argmax(results)
   tag = classes[results_index]


   # Retrieve response based on tag
   for intent in intents['intents']:
       if intent['tag'] == tag:
           responses = intent['responses']
           return random.choice(responses)


# Main code
while True:
   input_text = input("You: ")


   # Detect language using langid
   detected_lang = langid.classify(input_text)[0]


   # Map language codes to supported languages
   lang_map = {'en': 'English', 'hi': 'Hindi', 'sa': 'Sanskrit', 'fr': 'French'}


   if detected_lang in lang_map:
       print(f"Detected language: {lang_map[detected_lang]}")
       intents = load_intents(detected_lang)
       if intents:
           words = pickle.load(open('.venv/words.pkl', 'rb'))
           classes = pickle.load(open('.venv/classes.pkl', 'rb'))


           response = get_response(input_text, detected_lang)
           print("ChatBot:", response)
   else:
       print("Unsupported language")
