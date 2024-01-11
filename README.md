This projects attempts to make a Fake News Detection Model.
There will two outcomes as a result : FAKE or REAL
We do so using a TF-IDF Vectorizer and a PassiveAggressive Algorithm

TF (Term Frequency) : Number of times a word appears in a document. Higher the term frequency, higher the occurence of the word.
IDF (Inverse document frequency) : This gives an idea of how important a word might be across a whole corpus of documents.
Rare words have a higher value and a common word have a lower value.

PassiveAggressive Algorithm : It is an online learning algorithm. 
It remains PASSIVE for correct classification and AGGRESSIVE for mistakes/incorrect classification.
The algorithms doesn't converge. This means the data doesn't fit the model, due to the volume of poorely fitted observations.
Its purpose is to make updates that correct the loss, causing very little change in the norm of the weight vector.

Project Process
1.Import the dataset
2.Using sklearn, we build a TF-IDF vectorizer on the dataset
3.Initiliaze the PassiveAggressive Algorithm and fit the model ( training the model to gain insights)
4.The accuracy score and the confusion matrix tell us how well our model fares.
