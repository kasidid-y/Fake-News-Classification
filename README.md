# Fake News Classification 

build a a text classification model using Logistic Regression and evalute its performance.

**Tools and Environment in this project**

|    Tool/Environment	     |  Purpose                                 |                                                                              
|--------------------------|-----------------------------------------|
| Python	                 | Programming language for the project.   |
| Jupyter Notebook with Anaconda   |  Development environment for writing and running code.|
| NLTK   | Natural Language Processing tasks (tokenization, stopword removal) |  
|scikit-learn	                                    | Machine learning tasks (feature extraction, model training, evaluation). |
|Pandas	                                          | Data manipulation and analysis.                                                |     
|re                                               | Regular expressions for text cleaning.                                             |
|Prepared text dataset of news article from Kaggle           |Datset (a news articel with 35,028 real and 37,106 fake news) for training and testing model                                         |

 Dataset resource : [https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification](https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification)

**Key Steps**

Key Steps

- Text Preprocessing:

  Clean the text data (e.g., remove punctuation, stopwords, etc.).

  Tokenize the text into words or sentences using NLTK's word_tokenize.

- Custom Tokenization:

    Define a custom function (custom_word_tokenize) to tokenize text and remove stopwords.

- Vectorization:

    Convert the tokenized text into numerical features using TF-IDF Vectorization (TfidfVectorizer).

- Train-Test Split:

    Split the dataset into training and testing sets using train_test_split.

- Model Training:

    Train a Logistic Regression model on the training data.

- Model Evaluation:

    Predict labels for the test data.

    Evaluate the model using accuracy and a classification report.

    Create a confusion matrix to visualize the model's performance.
