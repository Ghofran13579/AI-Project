# -*- coding: utf-8 -*-
"""
Created on Wed May  1 01:16:32 2024

@author: massa
"""

import speech_recognition as sr
import pyttsx3
import random

class VoiceChatbot:
    def __init__(self, corpus):
        self.responses = corpus.split('\n')
        self.engine = pyttsx3.init()
        self.recognizer = sr.Recognizer()
        
    def generate_response(self, input_text):
        return random.choice(self.responses)
    
    def speak_response(self, response):
        self.engine.say(response)
        self.engine.runAndWait()
        
    def recognize_speech(self):
        with sr.Microphone() as source:
            print("Please speak something...")
            self.engine.say("Please speak something")
            self.engine.runAndWait()
            self.recognizer.adjust_for_ambient_noise(source)
            audio = self.recognizer.listen(source)
            try:
                text = self.recognizer.recognize_google(audio)
                print("You said:", text)
                return text
            except sr.UnknownValueError:
                print("Sorry, I couldn't understand what you said.")
                return ""
    
    def chat(self):
        while True:
            user_input = self.recognize_speech()
            if not user_input:
                continue
            if user_input.lower() == 'exit':
                print("Goodbye!")
                self.engine.say("Goodbye!")
                self.engine.runAndWait()
                break
            response = self.generate_response(user_input)
            print("Bot:", response)
            self.speak_response(response)

# Provide your own corpus or use a default one
corpus = """
Hello!
How are you?
I'm doing well, thank you.
What's your favorite color?
Blue is my favorite color.
"""

# Create a VoiceChatbot instance
chatbot = VoiceChatbot(corpus)

# Start the chat
chatbot.chat()
