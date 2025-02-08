# Fake News Classification 

build a text classification model using Logistic Regression and Naive Bayes model and evalute its performance.

# Tools and Environment in this project

|    Tool/Environment	     |  Purpose                                 |                                                                              
|--------------------------|-----------------------------------------|
| Python	                 | Programming language for the project.   |
| Jupyter Notebook with Anaconda   |  Development environment for writing and running code.|
| NLTK   | Natural Language Processing tasks (tokenization, stopword removal) |  
|scikit-learn	                                    | Machine learning tasks (feature extraction, model training, evaluation). |
|Pandas	                                          | Data manipulation and analysis.                                                |     
|re                                               | Regular expressions for text cleaning.                                             |
|Prepared text dataset of news article from Kaggle           |Datset for training and testing model                                         |

 Dataset resource : [https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification](https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification)

# Key Steps

- Text Preprocessing:

  Clean the text data using re(Regular expression) to remove special characters, download and import stopwords corpus from ntlk to remove any stopwords in text and lemmatizing words using WordNetLemmatizer from nltk.

- Feature Extraction

  Convert the text data into numerical features with TF-IDF Vectorization technique using TfidfVectorizer from sckit-learn. Building a set of unique words and paired words of a maximum of 5000 words 

- Split Data for Train and Test:

    Split the dataset into 5 set of training and testing sets using StratifiedKFold from scikit-learn.

- Model Training:

    Train a Logistic Regression model and Naive Bayes model on the training dataset using LogisticRegression and MultinomialNB from scikit-learn.
    

- Model Evaluation:
 
    Evaluate the model using accuracy, precision, recall_score, f1_score and confusion matrix.

**Confusion Matrix of Logistic Regression Model**

![confusionMatrix_lr](/assets/lr_cm.png)

**Confusion Matrix of Naive Bayes Model**

![confusionMatrix_nb](/assets/nb_cm.png)

**Testing Both model with new article**

![test](/assets/testmodel.png)

