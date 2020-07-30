import ngram
import generate

path = r"data/quotes.txt"

with open(path, "rb") as f:
    # decoding the bytes into a string
    text = f.read().decode()
order = 9
length = 85
model = ngram.train_lm(text, order)

result = generate.generate_text(model, order, nletters=length)

print(result)

"""
test to speech, you need to pip install pyttsx3
"""

import pyttsx3

engine = pyttsx3.init()

voices = engine.getProperty('voices')
converter = pyttsx3.init() 

converter.setProperty('rate', 350) 

engine.setProperty('voice', voices[0].id) 
engine.say(result)
engine.runAndWait()