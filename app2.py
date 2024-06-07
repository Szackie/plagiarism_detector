from flask import Flask, request, render_template,send_file, make_response, session
import io
import base64
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg') 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import xlsxwriter
# from nltk.lm import MLE, WittenBellInterpolated
from nltk.util import ngrams, pad_sequence, everygrams
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')
import re
import os
from flask_session import Session
from datetime import timedelta

# TODO: dodanie możliwości wyboru modeli językowych MLE i WBI do porównania + wizualizacja perplexity
# TODO: chmurka tagów z najczęstszymi słowami w tekście
# TODO: Wykresy zależności miary podobieństwa od długości tekstu
# TODO: wydajność - porównanie czasu obliczeń dla różnych modeli językowych
# TODO: wykresy stosunku liczby słów w tekście do całego tekstu dla każdego słowa i wszystkich tekstów na jednym wykresie (histogram) + możliwość 
#       wyboru tekstu do porównania (dropdown lub radio buttons)  
# TODO: dodawanie i usuwanie plików do porównania dynamicznie
# TODO: dodanie okienek do wklejania porównywanych tekstów


class PlagiarismDetector:
    def __init__(self):
        self.file_data = {}
        self.similarities = None

        # TODO: dla porównania modeli językowych MLE i WBI
        # self.mle_models = None
        # self.wbi_models = None

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
        
        # TODO:dla porównania modeli językowych MLE i WBI
        # self.mle_models = [self.build_ngram_model(text, MLE) for text in self.tokenized_texts]
        # self.wbi_models = [self.build_ngram_model(text, WittenBellInterpolated) for text in self.tokenized_texts]
        self.similarities = [[self.calculate_similarity(text1, text2) for text2 in file_data.values()] for text1 in file_data.values()]
                
    def visualize_similarities(self):

        plt.imshow(self.similarities, cmap='Reds', interpolation='nearest', vmin=0, vmax=1)
        plt.xticks(range(len(list(self.file_data.keys()))), list(self.file_data.keys()), rotation=90)
        plt.yticks(range(len(list(self.file_data.keys()))), list(self.file_data.keys()))
        plt.colorbar()
        plt.gca().xaxis.tick_top()
        plt.tight_layout()
        # Save the plot to a BytesIO object
        img = io.BytesIO()
        plt.savefig(img, format='png',dpi = 80)
        img.seek(0)

        # Encode the BytesIO object to base64 and decode it to utf-8 to embed it in HTML
        self.plot_url = base64.b64encode(img.getvalue()).decode()
        plt.clf()

    def set_state(self, state):
        self.file_data = state['file_data']

    def get_state(self):
        return {
            'file_data': self.file_data,
        }

'''
    #TODO: Metoda do wizualizacji perplexity dla modeli MLE i WBI dla każdego tekstu
    
    # def visualize_perplexities(self):
    #     mle_perplexities = [[model.perplexity(text) for text in self.tokenized_texts] for model in self.mle_models]
    #     wbi_perplexities = [[model.perplexity(text) for text in self.tokenized_texts] for model in self.wbi_models]

    #     fig, axs = plt.subplots(2)

    #     axs[0].imshow(mle_perplexities, cmap='Reds', interpolation='nearest')
    #     axs[0].set_title('MLE Perplexities')
    #     axs[0].set_xticks(range(len(list(self.file_data.keys()))))
    #     axs[0].set_yticks(range(len(list(self.file_data.keys()))))
    #     axs[0].set_xticklabels(list(self.file_data.keys()), rotation=90)
    #     axs[0].set_yticklabels(list(self.file_data.keys()))

    #     axs[1].imshow(wbi_perplexities, cmap='Reds', interpolation='nearest')
    #     axs[1].set_title('WBI Perplexities')
    #     axs[1].set_xticks(range(len(list(self.file_data.keys()))))
    #     axs[1].set_yticks(range(len(list(self.file_data.keys()))))
    #     axs[1].set_xticklabels(list(self.file_data.keys()), rotation=90)
    #     axs[1].set_yticklabels(list(self.file_data.keys()))

    #     plt.tight_layout()
    #     plt.colorbar()

    #     # Save the plot to a BytesIO object
    #     img = io.BytesIO()
    #     plt.savefig(img, format='png')
    #     img.seek(0)

    #     # Encode the BytesIO object to base64 and decode it to utf-8 to embed it in HTML
    #     self.plot_url = base64.b64encode(img.getvalue()).decode()
    #     plt.clf()
'''

def get_or_restore_detector():
    session.permanent = True
    detector_data = session.get('detector_data')
    detector = PlagiarismDetector()
    if detector_data is None:
        detector_data = detector.get_state()
        session['detector_data'] = detector_data
    else:
        detector.set_state(detector_data)
    return detector


app = Flask(__name__)
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SECRET_KEY'] = os.urandom(24)
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(minutes=5)
Session(app)

@app.route('/', methods=['GET'])
def show_site():
    detector = get_or_restore_detector()

    return render_template('index.html', plot_url=detector.plot_url, list_files=list(detector.file_data.keys()))

@app.route('/upload', methods=['POST'])
def upload_file():
    detector = get_or_restore_detector()

    if 'files' not in request.files:
        return 'No file part'
    files = request.files.getlist('files')
    for file in files:
        if file.filename == '':
            return render_template('index.html', plot_url=detector.plot_url, list_files=list(detector.file_data.keys()))
        if file:
            file_content = file.read().decode('utf-8')
            detector.file_data[file.filename] = file_content
    return render_template('index.html', plot_url=detector.plot_url, list_files=list(detector.file_data.keys()))

@app.route('/visualize', methods=['POST'])
def show_png():
    detector = get_or_restore_detector()
    if detector.file_data:
        detector.compare_files(detector.file_data)
        detector.visualize_similarities()
    else:
        detector.plot_url = None

    # TODO:
    # detector.visualize_perplexities()

    return render_template('index.html', plot_url=detector.plot_url, list_files=list(detector.file_data.keys()))

@app.route('/delete/<filename>', methods=['DELETE'])
def delete_file(filename):
    detector = get_or_restore_detector()
    if filename in detector.file_data.keys():
        del detector.file_data[filename]
        return render_template('index.html', plot_url=detector.plot_url, list_files=list(detector.file_data.keys())), 204
    else:
        return render_template('index.html', plot_url=detector.plot_url, list_files=list(detector.file_data.keys())), 404

@app.route('/download', methods=['GET'])
def download_file():
    detector = get_or_restore_detector()

    detector.compare_files(detector.file_data)

    output = io.BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    
#get excel report
    df = pd.DataFrame(detector.similarities, columns=list(detector.file_data.keys()), index=list(detector.file_data.keys()))
    df.to_excel(writer, sheet_name='Plagiarism Report')

    writer.close()  
    output.seek(0)
    response = make_response(output.read())
    response.headers['Content-Disposition'] = 'attachment; filename=plagiarism_report.xlsx'
    response.headers['Content-type'] = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    return response


if __name__ == '__main__':
    app.run(host='0.0.0.0')