from flask import Flask, request, render_template,send_file, make_response
import io
import base64
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg') 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.lm import MLE, WittenBellInterpolated
from nltk.util import ngrams, pad_sequence, everygrams
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')
import re
import os
class PlagiarismDetector:
    def __init__(self):
        self.file_data = {}
        self.similarities = None

        # dla porównania modeli językowych MLE i WBI
        # self.mle_models = None
        # self.wbi_models = None

        self.list_names = None
        self.plot_url = None
        self.n = 4

    def preprocess_and_tokenize(self,text):
        text = text.lower()
        text = re.sub(r"\[.*\]|\{.*\}", "", text)
        text = re.sub(r'[^\w\s]', "", text)
        return list(pad_sequence(word_tokenize(text), self.n, pad_left=True, left_pad_symbol="<s>"))

    def build_ngram_model(self,data, model_type):
        model = model_type(self.n)
        model.fit([list(everygrams(data, max_len=self.n))], vocabulary_text=data)
        return model

    def calculate_similarity(self,text1, text2):
        vectorizer = TfidfVectorizer().fit_transform([text1, text2])
        return cosine_similarity(vectorizer)[0,1]

    def compare_files(self,file_data):
        self.tokenized_texts = [self.preprocess_and_tokenize(text) for text in file_data.values()]
        self.mle_models = [self.build_ngram_model(text, MLE) for text in self.tokenized_texts]
        self.wbi_models = [self.build_ngram_model(text, WittenBellInterpolated) for text in self.tokenized_texts]
        self.similarities = [[self.calculate_similarity(text1, text2) for text2 in file_data.values()] for text1 in file_data.values()]
        self.list_names = list(file_data.keys())
    
    def generate_excel_report(self, writer):
        df = pd.DataFrame(self.similarities, columns=self.list_names, index=self.list_names)
        df.to_excel(writer, sheet_name='Plagiarism Report')
        
    def visualize_similarities(self):
        plt.imshow(self.similarities, cmap='Reds', interpolation='nearest')
        plt.xticks(range(len(self.list_names)), self.list_names, rotation=90)
        plt.yticks(range(len(self.list_names)), self.list_names)
        plt.colorbar()

        # Save the plot to a BytesIO object
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)

        # Encode the BytesIO object to base64 and decode it to utf-8 to embed it in HTML
        self.plot_url = base64.b64encode(img.getvalue()).decode()
        plt.clf()

    # Metoda do wizualizacji perplexity dla modeli MLE i WBI dla każdego tekstu
    
    # def visualize_perplexities(self):
    #     mle_perplexities = [[model.perplexity(text) for text in self.tokenized_texts] for model in self.mle_models]
    #     wbi_perplexities = [[model.perplexity(text) for text in self.tokenized_texts] for model in self.wbi_models]

    #     fig, axs = plt.subplots(2)

    #     axs[0].imshow(mle_perplexities, cmap='Reds', interpolation='nearest')
    #     axs[0].set_title('MLE Perplexities')
    #     axs[0].set_xticks(range(len(self.list_names)))
    #     axs[0].set_yticks(range(len(self.list_names)))
    #     axs[0].set_xticklabels(self.list_names, rotation=90)
    #     axs[0].set_yticklabels(self.list_names)

    #     axs[1].imshow(wbi_perplexities, cmap='Reds', interpolation='nearest')
    #     axs[1].set_title('WBI Perplexities')
    #     axs[1].set_xticks(range(len(self.list_names)))
    #     axs[1].set_yticks(range(len(self.list_names)))
    #     axs[1].set_xticklabels(self.list_names, rotation=90)
    #     axs[1].set_yticklabels(self.list_names)

    #     plt.tight_layout()
    #     plt.colorbar()

    #     # Save the plot to a BytesIO object
    #     img = io.BytesIO()
    #     plt.savefig(img, format='png')
    #     img.seek(0)

    #     # Encode the BytesIO object to base64 and decode it to utf-8 to embed it in HTML
    #     self.plot_url = base64.b64encode(img.getvalue()).decode()
    #     plt.clf()

detector = PlagiarismDetector()
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    
    if request.method == 'POST':

        if 'files' not in request.files:
            return 'No file part'
        files = request.files.getlist('files')
        for file in files:
            if file.filename == '':
                return render_template('index.html')
            if file:
                file_content = file.read().decode('utf-8')
                detector.file_data[file.filename] = file_content
                
        detector.compare_files(detector.file_data)
        detector.visualize_similarities()
        # detector.visualize_perplexities()
        return render_template('index.html', plot_url=detector.plot_url)
    return render_template('index.html')


@app.route('/download', methods=['GET'])
def download_file():
    output = io.BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    detector.generate_excel_report(writer)
    writer.close()  
    output.seek(0)
    response = make_response(output.read())
    response.headers['Content-Disposition'] = 'attachment; filename=plagiarism_report.xlsx'
    response.headers['Content-type'] = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    return response


if __name__ == '__main__':
    app.run(host='0.0.0.0')