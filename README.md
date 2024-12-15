# Smart City Slicker

# Description
The term “smart city” is widely used, but there is no consensus on the definition. Many citizens and stakeholders are unsure about what a smart city means in their community and how it affects them. Imagine you are a stakeholder in a rising Smart City and want to know more about themes and concepts about existing smart cities. You also want to know where does your smart city place among others. In this project, you will use text analysis techniques to investigate themes and similarities for smart cities with the use of cluster analysis, topic modeling, and summarization. This project can help you as a stakeholder understand smart cities using data from the 2015 Smart City Challenge.

# How to install
Clone the repo

# How to run
```pipenv run python project3.py --document "city.pdf"``` the command for document has to be in quotations. ```pipenv run python -m pytest``` for test

# tree
.
├── COLLABORATORS
├── LICENSE
├── Pipfile
├── Pipfile.lock
├── README.md
├── __pycache__
│   └── contractions.cpython-310.pyc
├── contractions.py
├── docs
├── model.pkl
├── project3.ipynb
├── project3.py
├── setup.cfg
├── setup.py
├── smartcity
│   ├── AK Anchorage.pdf
│   ├── AL Birmingham.pdf
│   ├── AL Montgomery.pdf
│   ├── AZ Scottsdale AZ.pdf
│   ├── AZ Tucson.pdf
│   ├── CA Chula Vista.pdf
│   ├── CA Fremont.pdf
│   ├── CA Fresno.pdf
│   ├── CA Long Beach.pdf
│   ├── CA Moreno Valley.pdf
│   ├── CA Oakland.pdf
│   ├── CA Oceanside.pdf
│   ├── CA Riverside.pdf
│   ├── CA Sacramento.pdf
│   ├── CA San Jose_0.pdf
│   ├── CT NewHaven.pdf
│   ├── DC_0.pdf
│   ├── FL Jacksonville.pdf
│   ├── FL Miami.pdf
│   ├── FL Orlando.pdf
│   ├── FL St. Petersburg.pdf
│   ├── FL Tallahassee.pdf
│   ├── FL Tampa.pdf
│   ├── GA Atlanta.pdf
│   ├── GA Brookhaven.pdf
│   ├── GA Columbus.docx
│   ├── IA Des Moines.pdf
│   ├── IN Indianapolis.pdf
│   ├── KY Louisville.pdf
│   ├── LA Baton Rouge.pdf
│   ├── LA New Orleans.pdf
│   ├── LA Shreveport.pdf
│   ├── MA Boston.pdf
│   ├── MD Baltimore.pdf
│   ├── MI Detroit.pdf
│   ├── MI Port Huron and Marysville.pdf
│   ├── MN Minneapolis St Paul.pdf
│   ├── MO St. Louis.pdf
│   ├── NC Charlotte.pdf
│   ├── NC Greensboro.pdf
│   ├── NC Raleigh.pdf
│   ├── NE Lincoln.pdf
│   ├── NE Omaha.pdf
│   ├── NJ Jersey City.pdf
│   ├── NJ Newark.pdf
│   ├── NM Albuquerque.docx
│   ├── NV Las Vegas.pdf
│   ├── NV Reno.pdf
│   ├── NY Albany Troy Schenectady Saratoga Springs.pdf
│   ├── NY Buffalo.pdf
│   ├── NY Mt Vernon Yonkers New Rochelle.pdf
│   ├── NY Rochester.pdf
│   ├── OH Akron.pdf
│   ├── OH Canton.pdf
│   ├── OH Cleveland.pdf
│   ├── OH Toledo.pdf
│   ├── OK Oklahoma City.pdf
│   ├── OK Tulsa.pdf
│   ├── RI Providence.pdf
│   ├── SC Greenville.pdf
│   ├── TN Chattanooga.pdf
│   ├── TN Memphis.pdf
│   ├── TN Nashville.pdf
│   ├── TX Lubbock.pdf
│   ├── VA Newport News.pdf
│   ├── VA Norfolk.pdf
│   ├── VA Richmond.pdf
│   ├── VA Virginia Beach.pdf
│   ├── WA Seattle.pdf
│   ├── WA Spokane.pdf
│   └── WI Madison.pdf
├── smartcity_eda.tsv
├── smartcity_predict.tsv
└── tests
    └── test_project3.py


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

