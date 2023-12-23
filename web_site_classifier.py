import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

def cleaning_processing(sentence):

    stop_words = set(stopwords.words('english')) 
    stemmer = WordNetLemmatizer()
    cleaned_sentence = []
    
    words = word_tokenize(sentence)                     # Split the sentence into words
    for word in words:
        word = word.lower()                             # Convert to lowercase
        word = re.sub(r"[^a-zA-Z]", "", word)           # Remove special characters, numbers, punctuation, HTML tags

        if word not in stop_words and word != '':  	# Remove empty string and remove stop words
            word = stemmer.lemmatize(word)  		# Lemmatization - convert into base root word
            cleaned_sentence.append(word)

    filtered_sentence = " ".join(cleaned_sentence)  	# Convert the list of words into a string

    return filtered_sentence

def main():
    # Read Data
    data = pd.read_csv("website_classification.csv")
    print(data.head())

    # Drop unwanted columns in data
    data = data.drop(["Unnamed: 0", "website_url"], axis=1)
    print("Remove unwanted colums")
    print(data.head())

    # Remove duplicate values in data
    data = data.drop_duplicates()

    # Set Feature and Target Values
    feature= data['cleaned_website_text']
    target = data['Category']

    # Cleaning the feature column
    feature= feature.apply(cleaning_processing)
    
    # Splitting the Data Train, Test Split
    x_train, x_test, y_train, y_test = train_test_split(feature, target, test_size=0.2, random_state=42)
    
    #Check the shape of feature and target
    print("Before cheking the shape")
    print("X_train shape:", x_train.shape)
    print("X_test shape:", x_test.shape)
    print("y_train shape:", y_train.shape) 
    print("y_test shape:", y_test.shape)
    
    # Convert word into vector
    vectorizer = TfidfVectorizer(binary=True)
    x_train_tfidf = vectorizer.fit_transform(x_train)
    x_test_tfidf = vectorizer.transform(x_test)
    
    # After applying vectorizer check shape of feture and target
    print("After checking the shpe-vecorizer")
    print("x_train_tfidf:", x_train_tfidf.shape)
    print("x_test_tfidf:", x_test_tfidf.shape)
    print("y_train:", y_train.shape)
    print("y_test:", y_test.shape)
    
    # Model Building
    classifier = KNeighborsClassifier()
    classifier.fit(x_train_tfidf, y_train)
    y_predict = classifier.predict(x_test_tfidf)
    accuracy = accuracy_score(y_test, y_predict)
    
    # Evaluation Metrics
    print("KNeighborsClassifier")
    print("KNeighborsClassifier accuracy score: ", accuracy)
    print(confusion_matrix(y_test, y_predict))
    print("Predicted Values")
    #print(y_predict)

if __name__ == "__main__":
    main()
    
    
    
