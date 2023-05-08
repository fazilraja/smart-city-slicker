import PyPDF2
import os
import pandas as pd
import numpy as np
import nltk
import spacy
import unicodedata
from contractions import CONTRACTION_MAP
import re
from nltk.corpus import wordnet
import collections
#from textblob import Word
from nltk.tokenize.toktok import ToktokTokenizer
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from joblib import dump, load
import argparse
import csv
K = 36

def load_data(filename):
    
    if filename.endswith(".pdf"):
        with open(filename, 'rb') as pdfFileObj:
            pdfReader = PyPDF2.PdfReader(pdfFileObj)
            num_pages = len(pdfReader.pages)
            text = ""
            for i in range(num_pages):
                page = pdfReader.pages[i]
                page_text = page.extract_text()
                page_lines = page_text.splitlines()
                text += page_text

    return {text}

def normalize(files_data):
    #nltk.download('stopwords')

    tokenizer = ToktokTokenizer()
    stopword_list = nltk.corpus.stopwords.words('english')
    nlp = spacy.load('en_core_web_sm', exclude=['parser'])
    # nlp_vec = spacy.load('en_vectors_web_lg', parse=True, tag=True, entity=True)

    def strip_html_tags(text):
        soup = BeautifulSoup(text, "html.parser")
        if bool(soup.find()):
            [s.extract() for s in soup(['iframe', 'script'])]
            stripped_text = soup.get_text()
            stripped_text = re.sub(r'[\r|\n|\r\n]+', '\n', stripped_text)
        else:
            stripped_text = text
        return stripped_text


    #def correct_spellings_textblob(tokens):
    #	return [Word(token).correct() for token in tokens]  


    def simple_porter_stemming(text):
        ps = nltk.porter.PorterStemmer()
        text = ' '.join([ps.stem(word) for word in text.split()])
        return text


    def lemmatize_text(text):
        text = nlp(text)
        text = ' '.join([word.lemma_ if word.lemma_ != '-PRON-' else word.text for word in text])
        return text


    def remove_repeated_characters(tokens):
        repeat_pattern = re.compile(r'(\w*)(\w)\2(\w*)')
        match_substitution = r'\1\2\3'
        def replace(old_word):
            if wordnet.synsets(old_word):
                return old_word
            new_word = repeat_pattern.sub(match_substitution, old_word)
            return replace(new_word) if new_word != old_word else new_word
                
        correct_tokens = [replace(word) for word in tokens]
        return correct_tokens


    def expand_contractions(text, contraction_mapping=CONTRACTION_MAP):
        
        contractions_pattern = re.compile('({})'.format('|'.join(contraction_mapping.keys())), 
                                        flags=re.IGNORECASE|re.DOTALL)
        def expand_match(contraction):
            match = contraction.group(0)
            first_char = match[0]
            expanded_contraction = contraction_mapping.get(match)\
                                    if contraction_mapping.get(match)\
                                    else contraction_mapping.get(match.lower())                       
            expanded_contraction = first_char+expanded_contraction[1:]
            return expanded_contraction
            
        expanded_text = contractions_pattern.sub(expand_match, text)
        expanded_text = re.sub("'", "", expanded_text)
        return expanded_text


    def remove_accented_chars(text):
        text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        return text


    def remove_special_characters(text, remove_digits=False):
        pattern = r'[^a-zA-Z0-9\s]|\[|\]' if not remove_digits else r'[^a-zA-Z\s]|\[|\]'
        text = re.sub(pattern, '', text)
        return text


    def remove_stopwords(text, is_lower_case=False, stopwords=stopword_list):
        tokens = tokenizer.tokenize(text)
        tokens = [token.strip() for token in tokens]
        if is_lower_case:
            filtered_tokens = [token for token in tokens if token not in stopwords]
        else:
            filtered_tokens = [token for token in tokens if token.lower() not in stopwords]
        filtered_text = ' '.join(filtered_tokens)    
        return filtered_text

    def remove_cities_states(text) :
        labels = ['GPE', 'LOC', 'PERSON', 'ORG']

        cities_states = ['Brookhaven', 'Tallahassee', 'Buffalo', 'Riverside', 'Scottsdale', 'Jacksonville', 'New Orleans', 'Montgomery', 'Port Huron', 'Marysville', 'Seattle', 'Shreveport', 'Spokane', 'Indianapolis', 'Birmingham', 'Baton Rouge', 'Miami', 'Oceanside', 'San Jose', 'Lincoln', 'Boston', 'Sacramento', 'Richmond', 'Atlanta', 'Rochester', 'Memphis', 'Raleigh', 'Albany', 'Troy', 'Schenectady', 'Saratoga Springs', 'Cleveland', 'Charlotte', 'Jersey City', 'Chula Vista', 'Long Beach', 'Detroit', 'Des Moines', 'St. Louis', 'Omaha', 'Akron', 'Newport News', 'Mt Vernon', 'Yonkers', 'New Rochelle', 'Fremont', 'Baltimore', 'Greenville', 'NewHaven', 'Lubbock', 'Fresno', 'Oakland', 'Chattanooga', 'Providence', 'Anchorage', 'Tucson', 'Minneapolis', 'Reno', 'Toledo', 'Greensboro', 'Canton', 'Las Vegas', 'Nashville', 'Oklahoma City', 'Madison', 'Newark', 'Louisville', 'St. Petersburg', 'Moreno Valley', 'Tampa', 'Norfolk', 'Washington, DC', 'Orlando', 'Virginia Beach', 'Tulsa']

        doc = nlp(text)

        for ent in doc.ents:
            if ent.label_ in labels or ent.text in cities_states:
                text = text.replace(ent.text, '')

        return text


    def normalize_corpus(corpus,html_stripping=True, contraction_expansion=True,
                        accented_char_removal=True, text_lower_case=True, 
                        text_stemming=False, text_lemmatization=True, 
                        special_char_removal=True, remove_digits=True,
                        stopword_removal=True, stopwords=stopword_list, cites_states = True):
        
        normalized_corpus = []
        # normalize each document in the corpus
        for doc in corpus:
            # strip HTML
            if html_stripping:
                doc = strip_html_tags(doc)
            
            # remove extra newlines
            doc = doc.translate(doc.maketrans("\n\t\r", "   "))

            # remove states and cities
            if cites_states:
                doc = remove_cities_states(doc)
                
            # remove accented characters
            if accented_char_removal:
                doc = remove_accented_chars(doc)

            # expand contractions    
            if contraction_expansion:
                doc = expand_contractions(doc)

            # lemmatize text
            if text_lemmatization:
                doc = lemmatize_text(doc)

            # stem text
            if text_stemming and not text_lemmatization:
                doc = simple_porter_stemming(doc)

            # remove special characters and\or digits  
            if special_char_removal:
                # insert spaces between special characters to isolate them    
                special_char_pattern = re.compile(r'([{.(-)!}])')
                doc = special_char_pattern.sub(" \\1 ", doc)
                doc = remove_special_characters(doc, remove_digits=remove_digits)  

            # remove extra whitespace
            doc = re.sub(' +', ' ', doc)

            # lowercase the text    
            if text_lower_case:
                doc = doc.lower()

            # remove stopwords
            if stopword_removal:
                doc = remove_stopwords(doc, is_lower_case=text_lower_case, stopwords=stopwords)

            # remove extra whitespace
            doc = re.sub(' +', ' ', doc)
            doc = doc.strip()
            normalized_corpus.append(doc)
        return normalized_corpus
    
    return normalize_corpus(files_data)

def load_model():
    with open('model.pkl', 'rb') as f:
        model = load(f)
    return model    

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Project 3')
    parser.add_argument('--document', required=True, help='Document to predict')

    args = parser.parse_args()

    files_data = load_data(args.document)

    # normalize corpus
    normalized_corpus = normalize(files_data)  

    # load model
    model = load_model()  

    # create vectors
    vectorizer = TfidfVectorizer(max_features=20353)
    X = vectorizer.fit_transform(normalized_corpus)
    
    # predict
    kmeans = KMeans(n_clusters=1, random_state=0).fit(X)

    # get cluster id
    files_cluster_data = kmeans.predict(X)

    # print results
    with open("smartcity_predict.tsv", "w") as f:
        writer = csv.writer(f, delimiter="\t")

        writer.writerow(["city", "raw text", "clean text", "cluster id"])

        writer.writerow([args.document, files_data, normalized_corpus, files_cluster_data])


