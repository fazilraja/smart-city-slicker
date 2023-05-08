# cs5293sp23-project3

# Description
The term “smart city” is widely used, but there is no consensus on the definition. Many citizens and stakeholders are unsure about what a smart city means in their community and how it affects them. Imagine you are a stakeholder in a rising Smart City and want to know more about themes and concepts about existing smart cities. You also want to know where does your smart city place among others. In this project, you will use text analysis techniques to investigate themes and similarities for smart cities with the use of cluster analysis, topic modeling, and summarization. This project can help you as a stakeholder understand smart cities using data from the 2015 Smart City Challenge.

# How to install
Clone the repo

# How to run
pipenv run python project3.py --document city.pdf 

# External libraries
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

# Code organization

## load data
Loads the pdf given and converts it to text

## normalizer
Using the textbook code it normalizes and cleans the code

## load model
loads the model used in Part 1

## main
Gets arguments and runs the code

# Assumptions
The model loaded but I could not fit the model correctly due to the features not being the same size. The trained model was of size 20353 while the test tfidf vector only had 1680 features. 
