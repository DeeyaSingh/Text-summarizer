from flask import Flask, render_template, request
app = Flask(__name__)
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch

class Summarizer:
    def __init__(self):
        self.model = T5ForConditionalGeneration.from_pretrained("t5-large")
        self.tokenizer = T5Tokenizer.from_pretrained("t5-large")

    def summarize(self, text: str, max_length: int) -> str:
        # Tokenize the input text
        input_ids = self.tokenizer.encode("summarize: " + text, return_tensors="pt", truncation=True)

        summary_max_length = 400

        with torch.no_grad():
            summary_ids = self.model.generate(input_ids, max_length=summary_max_length, num_beams=4, length_penalty=2.5, min_length=int(0.2 * summary_max_length), early_stopping=True)

        
        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary

@app.route('/', methods=['GET'])
def index():
    
    default_text = """Deep Learning is a subfield of machine learning that deals with algorithms inspired by the structure and function of the brain called artificial neural networks. At its core, deep learning seeks to automatically learn data representations without manual feature engineering.

Historically, neural networks have been around for decades, but they gained immense popularity in the 21st century due to the availability of vast amounts of data and powerful computational resources, particularly Graphics Processing Units (GPUs). These factors allowed researchers and practitioners to build larger and deeper networks, leading to significant advancements in various tasks such as image and speech recognition, machine translation, and game playing.

The architecture of deep learning models is layered, with each layer processing the data and passing it to the next. These layers can recognize patterns with increasing levels of abstraction. For example, in image recognition, initial layers might recognize edges, middle layers could recognize shapes, and deeper layers could recognize complex structures.

Key deep learning models include Convolutional Neural Networks (CNNs) for tasks related to image data, Recurrent Neural Networks (RNNs) for sequential data, and Transformers, which have set state-of-the-art performance in natural language processing tasks.

While deep learning has led to numerous breakthroughs, it also poses challenges. The models often require large amounts of data and compute resources, and their decision-making processes can be hard to interpret, leading to concerns about transparency and fairness."""
    default_summary = get_summary(default_text)  
    return render_template('index.html', default_text=default_text, default_summary=default_summary)

@app.route('/summarize', methods=['POST'])
def summarize_text():
    text = request.form['text_to_summarize']
    summary = get_summary(text)
    return render_template('index.html', default_text=text, default_summary=summary) 

def get_summary(text):
    summarizer = Summarizer()
    max_length = 400
    return summarizer.summarize(text, max_length=max_length)









if __name__ == '__main__':
    app.run(debug=True)
