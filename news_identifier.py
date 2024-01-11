#Categorizing news as REAL or FAKE
#We do so using a TF-IDF Vectorizer and a PassiveAggressive Algorithm

#TF (Term Frequency) : Number of times a word appears in a document. Higher the term frequency, higher the occurence of the word.
#IDF (Inverse document frequency) : This gives an idea of how important a word might be across a whole corpus.
#Rare words that have a higher value and a common word has a low value.

#PassiveAggressive Algorithm : It is an online learning algorithm. 
#It remains passive for PASSIVE  for correct classification and AGGRESIVE for mistakes/incorrect classification.
#The algorithms doesn't converge. This means the data doesn't fit the model, due to the volume of poorely fitted observations.
#Its purpose is to make updates that correct the loss, causing very little change in the norm of the weight vector.

#Scaling is important so that the data is in a consistent range of values

#Project Process
#1.Import the dataset
#2.Using sklearn, we build a TF-IDF vectorizer on the dataset
#3.Initiliaze the PassiveAggressive Algorithm and fit the model ( training the model to gain insights)
#4.The accuracy score and the confusion matrix tell us how well our model fares.


#Import the necessary libraries
import pandas as pd
import numpy as np
import itertools
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score,confusion_matrix

#1. The dataset we shall be dealing with
df=pd.read_csv(r'D:\Allen Archive\Allen Archives\NEU_academics\Semester1\Python_notebooks\Semester 1\Self_projects\News\news.csv')
#df.shape #df.head() #print(df)

#2.Splitting the dataset into training and testing sets
x_train,x_test,y_train,y_test = train_test_split(df['text'],df['label'],test_size=0.2,random_state=7)
#Random state is important for reproducing the same output across different times and for others to do the same
# Setting the random_state parameter to a fixed value ensures that the random splitting process will be the same every time you run the code and output remains same

#3.Training and transforming the dataset
#We will have to filter out Stop-words
#We shall be keeping a max term frequency of 0.7, anything higher is not considered.
#Initiliazing the TF-IDF vectorizer
tfidf_vectorizer=TfidfVectorizer(stop_words='english',max_df=0.7)
#Fit and transforming data
#Fit finds the mean and median
#Transform utlises mean and median from fit and applies scaling
tfidf_train=tfidf_vectorizer.fit_transform(x_train)
tfidf_test=tfidf_vectorizer.transform(x_test)

#4.Initiliazing the PassiveAggressive Algorithm
pass_agg_class = PassiveAggressiveClassifier(max_iter=50)
pass_agg_class.fit(tfidf_train,y_train)

#5.Finding scores to determine accuracy, for the testing data
y_predict=pass_agg_class.predict(tfidf_test)
score=accuracy_score(y_test,y_predict)
print(f'The accuracy score is {round(score*100,2)}%')

#Confusion matrix give us True/False positives and True/False negatives
con_mat = confusion_matrix(y_test,y_predict,labels=['FAKE','REAL'])
print(con_mat)