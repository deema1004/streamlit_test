# This file creates the 'pipe' NLP model and saves it as model.joblib

# Import libraries
import pandas as pd
import joblib
import os

from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from utils import preprocessor

tfidf = TfidfVectorizer()
classifier = LinearSVC()

print("Current Working Directory:", os.getcwd())

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_file_path = os.path.join(script_dir, 'sentiments.csv')
joblib_file_path = os.path.join(script_dir, 'model.joblib')
# print(csv_file_path)
# print(joblib_file_path)



if __name__ == "__main__":
   # may need to change the following to your location of sentiments.csv
   df = pd.read_csv(csv_file_path)  
   pipe = make_pipeline(preprocessor(), tfidf, classifier)
   pipe.fit(df['text'],df['sentiment'])
   joblib.dump(pipe, open(joblib_file_path,'wb'))