import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from keras.models import load_model
from googletrans import Translator

lemmatizer = WordNetLemmatizer()
model = load_model('.venv/chatbot_motivational.h5')
translator = Translator()

def load_intents(language):
   filename = f'intents_{language}.json'
   try:
       with open(filename, 'r', encoding='utf-8') as file:
           intents = json.load(file)
       return intents
   except FileNotFoundError:
       print(f"Intents file for {language} not found.")
       return None

def get_greeting_message(language):
    filename = f'intents_{language}.json'
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            intents = json.load(file)
        for intent in intents['intents']:
            if intent['tag'] == "greeting":
                responses = intent['patterns']
                return random.choice(responses)
    except FileNotFoundError:
        print(f"Intents file for {language} not found.")
        return None

def preprocess_input(input_text):
   tokens = nltk.word_tokenize(input_text)
   lemmatized_tokens = [lemmatizer.lemmatize(word.lower()) for word in tokens]
   return lemmatized_tokens


def get_response(input_text, lang):
   input_text = input_text.lower()
   processed_input = preprocess_input(input_text)


   if lang != 'en':
       translated_text = translator.translate(input_text, src=lang, dest='en').text
       processed_input = preprocess_input(translated_text)


   bag = [0] * len(words)
   for w in processed_input:
       for i, word in enumerate(words):
           if word == w:
               bag[i] = 1


   results = model.predict(np.array([bag]))[0]
   results_index = np.argmax(results)
   tag = classes[results_index]


   for intent in intents['intents']:
       if intent['tag'] == tag:
           responses = intent['responses']
           return {"tag": tag,"response":random.choice(responses)}


if __name__=="__main__":
    print("Name")
    name=input()
    print("Send 1. English 2. Hindi 3. Sanskrit 4. French")
    language_input = input()
    if(language_input.lower() == "english"):
        language_code = "en"
    elif(language_input.lower() == "hindi"):
        language_code = "hi"
    elif (language_input.lower() == "sanskrit"):
        language_code = "sa"
    elif (language_input.lower() == "french"):
        language_code = "fr"
    else:
        print("Language not supported")
    lang_map = {'en': 'English', 'hi': 'Hindi', 'sa': 'Sanskrit', 'fr': 'French'}

    if language_code in lang_map:
        intents = load_intents(language_code)
        if intents:
            words = pickle.load(open('.venv/words.pkl', 'rb'))
            classes = pickle.load(open('.venv/classes.pkl', 'rb'))

            response = get_response(get_greeting_message(language_code), language_code)
            print("ChatBot:", response["response"].format(name))

    while True:
        print(name,":")
        input_text = input()
        response = get_response(input_text, language_code)
        print("ChatBot:", response["response"].format(name))
        if(response["tag"]=="goodbye"):
            break
