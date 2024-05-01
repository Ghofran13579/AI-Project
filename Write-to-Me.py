# -*- coding: utf-8 -*-
"""
Created on Wed May  1 00:57:27 2024

@author: massa
"""

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load pre-trained model and tokenizer
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Define function to generate response
def generate_response(prompt, max_length=50, temperature=0.7):
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    output = model.generate(input_ids, max_length=max_length, temperature=temperature, num_return_sequences=1)
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response

# Main loop
while True:
    user_input = input("You: ")
    if user_input.lower() == 'exit':
        print("Chatbot: Goodbye!")
        break
    else:
        response = generate_response(user_input)
        print("Chatbot:", response)
