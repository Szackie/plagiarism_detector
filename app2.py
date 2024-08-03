from matplotlib.ticker import FuncFormatter
from flask import Flask, request, render_template,send_file, make_response, session
import io
from wordcloud import WordCloud
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
import numpy as np
from PIL import Image
from collections import Counter

# TODO: dodanie możliwości wyboru modeli językowych MLE i WBI do porównania + wizualizacja perplexity
# TODO: chmurka tagów z najczęstszymi słowami w tekstach (niech dla każdego z plików będzie inny kolor, słowa półprzeźroczyste i niech zachodzą na siebie)
# TODO: Wykresy zależności miary podobieństwa od długości tekstu
# TODO: wydajność - porównanie czasu obliczeń dla różnych modeli językowych
# TODO: wykresy stosunku liczby słów w tekście do całego tekstu dla każdego słowa i wszystkich tekstów na jednym wykresie (histogram) + możliwość 
#       wyboru tekstu do porównania (dropdown lub radio buttons)  
# TODO: [zrobione] dodawanie i usuwanie plików do porównania dynamicznie
# TODO: dodanie okienek do wklejania porównywanych tekstów


class WordCloudGenerator:
    def __init__(self, file_data=None):

        if file_data is None:
            file_data = {}
        self.file_data = file_data

    def generate_word_clouds(self):
        colors = ['#FF6F61', '#6B5B95', '#88B04B', '#F7CAC9', '#92A8D1', '#955251', '#B565A7', '#009B77', '#DD4124', '#45B8AC']
        
        cloud_url_list=[]
        for idx, (filename, text) in enumerate(self.file_data.items()):
            wordcloud = WordCloud(
                width=800,
                height=800,
                background_color=None,
                mode='RGBA',
                color_func=lambda *args, **kwargs: colors[idx % len(colors)],  
                prefer_horizontal=1.0, 
                max_words=20,
                
            ).generate(text)
            
            wordcloud_image = wordcloud.to_array()

            plt.figure(figsize=(8, 8))
            plt.title(filename)
            plt.imshow(wordcloud_image, interpolation='bilinear')
            plt.axis('off')
            
            img2 = io.BytesIO()
            plt.savefig(img2, format='png', dpi=80)
            img2.seek(0)

            cloud_url = base64.b64encode(img2.getvalue()).decode()
            plt.clf()
            cloud_url_list.append(cloud_url)
            plt.close()

        return cloud_url_list

class PlagiarismDetector:
    def __init__(self):
        self.file_data = {}
        self.similarities = None

        # TODO: dla porównania modeli językowych MLE i WBI
        # self.mle_models = None
        # self.wbi_models = None
        self.word_freq_plot_url = None
        self.plot_url = None
        self.cloud_url_list = None
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
    
    def visualize_word_frequencies(self):
        fig, ax = plt.subplots()
        colors = ['#FF6F61', '#6B5B95', '#88B04B', '#F7CAC9', '#92A8D1', '#955251', '#B565A7', '#009B77', '#DD4124', '#45B8AC']
        markers = ['+','x','d','*','1','o','.','s','d',',']

        for idx, (filename, content) in enumerate(self.file_data.items()):
            content = re.sub(r'[^\w\s]', '', content)
            words = re.split(r'\s+', content)
            word_counts = Counter(words)
            most_common_words = word_counts.most_common(10)
            words, counts = zip(*most_common_words)
            ax.scatter(words, counts, color=colors[idx % len(colors)],marker=markers[idx%len(markers)], label=filename)

        ax.set_xlabel('Words',color='Grey')
        ax.set_ylabel('Frequency',color='Grey')
        ax.legend()
        ax.grid()
                
        for label in ax.get_xticklabels():
            label.set_ha('right')
        plt.xticks(rotation=45)
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        self.word_freq_plot_url = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close(fig)

    def visualize_similarities(self):
        _, ax = plt.subplots()

        im = ax.imshow(self.similarities, cmap='Reds', interpolation='nearest', vmin=0, vmax=1)

        ax.set_xticks(range(len(list(self.file_data.keys()))))
        ax.set_xticklabels(list(self.file_data.keys()), rotation=90)
        ax.set_yticks(range(len(list(self.file_data.keys()))))
        ax.set_yticklabels(list(self.file_data.keys()))
        ax.xaxis.tick_top()

        def format_func(value, tick_number):
            if value == 0:
                return 'Low'
            elif value == 1:
                return 'High'
            else:
                return ''

        formatter = FuncFormatter(format_func)
        cbar = plt.colorbar(im, format=formatter)
        cbar.set_label('Similarity')  
        plt.tight_layout()

        img = io.BytesIO()
        plt.savefig(img, format='png', dpi=80)
        img.seek(0)

        self.plot_url = base64.b64encode(img.getvalue()).decode()
        plt.clf()
        
        generator = WordCloudGenerator(self.file_data)
        self.cloud_url_list = generator.generate_word_clouds()
        self.visualize_word_frequencies()
        plt.close()

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

    return render_template('index.html',cloud_url_list = detector.cloud_url_list, plot_url=detector.plot_url, 
                           word_freq_plot_url=detector.word_freq_plot_url,  list_files=list(detector.file_data.keys()))

@app.route('/upload', methods=['POST'])
def upload_file():
    detector = get_or_restore_detector()

    if 'files' not in request.files:
        return 'No file part'
    files = request.files.getlist('files')
    for file in files:
        if file.filename == '':
            return render_template('index.html',cloud_url_list=detector.cloud_url_list, plot_url=detector.plot_url, 
                           word_freq_plot_url=detector.word_freq_plot_url,  list_files=list(detector.file_data.keys()))
        if file:
            file_content = file.read().decode('utf-8')
            detector.file_data[file.filename] = file_content
    return render_template('index.html',cloud_url_list=detector.cloud_url_list, plot_url=detector.plot_url, 
                           word_freq_plot_url=detector.word_freq_plot_url,  list_files=list(detector.file_data.keys()))

@app.route('/visualize', methods=['POST'])
def show_png():
    detector = get_or_restore_detector()
    if detector.file_data:
        detector.compare_files(detector.file_data)
        detector.visualize_similarities()

    else:
        detector.word_freq_plot_url = None
        detector.plot_url = None
        detector.cloud_url_list = None

    # TODO:
    # detector.visualize_perplexities()

    return render_template('index.html',cloud_url_list=detector.cloud_url_list, plot_url=detector.plot_url, 
                           word_freq_plot_url=detector.word_freq_plot_url,  list_files=list(detector.file_data.keys()))

@app.route('/delete/<filename>', methods=['DELETE'])
def delete_file(filename):
    detector = get_or_restore_detector()
    if filename in detector.file_data.keys():
        del detector.file_data[filename]
        return render_template('index.html',cloud_url_list=detector.cloud_url_list, plot_url=detector.plot_url, 
                           word_freq_plot_url=detector.word_freq_plot_url,  list_files=list(detector.file_data.keys())), 204
    else:
        return render_template('index.html',cloud_url_list=detector.cloud_url_list, plot_url=detector.plot_url, 
                           word_freq_plot_url=detector.word_freq_plot_url,  list_files=list(detector.file_data.keys())), 404

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