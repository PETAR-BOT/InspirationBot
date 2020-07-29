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