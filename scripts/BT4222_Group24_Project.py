import re
import pandas as pd
import numpy as np
import gensim.downloader as api
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# # to create shortcut to Drive Folder

# import os, sys
# from google.colab import drive
# drive.mount('/content/drive')
# nb_path = '/content/notebooks'
# os.symlink('/content/drive/My Drive/Colab Notebooks/bt4222', nb_path)
# sys.path.insert(0,nb_path)

# !cp '/content/drive/My Drive/Colab Notebooks/bt4222/Reviews.csv' .

df = pd.read_csv('Reviews.csv', delimiter=",")

import nltk
from nltk import pos_tag
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer, SnowballStemmer
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
wordnet_lemmatizer = WordNetLemmatizer()
english_stopwords = stopwords.words('english')

def preprocess_data(d):
    # tokenize
    # nltk tokenizer (based on PunktSentenceTokenizer and TreebankWordTokenizer)
    tokens = [w for s in sent_tokenize(d) for w in word_tokenize(s)]

    tokens = [wordnet_lemmatizer.lemmatize(t.lower()) for t in tokens]
    tokens = [t.lower() for t in tokens]

    # remove stopwords
    # tokens = [t for t in tokens if t not in english_stopwords]
    return tokens

train = df[['Text', 'Score']]

train['Score'].value_counts()

from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler

ros = RandomUnderSampler({1:25000,2:25000,3:25000,4:25000,5:25000}, random_state=0)
train_underSample, _ = ros.fit_resample(train, train['Score'])

train_underSample['x'] = train_underSample['Text'].apply(lambda s: preprocess_data(s))
train_underSample['y'] = train_underSample['Score'].apply(lambda s: 1 if s > 3 else 0)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(train_underSample[['x']], train_underSample['y'], random_state=1, test_size=0.2)

X_test = X_test['x']
X_train = X_train['x']

"""# BoW + LR, BoW + MLP"""

# BoW
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()

bow_X_train = cv.fit_transform(X_train.apply(lambda s: ' '.join(s)))
bow_X_test = cv.transform(X_test.apply(lambda s: ' '.join(s)))

# BoW + LR
model = LogisticRegression(max_iter=10000)
model.fit(bow_X_train, y_train)
y_pred = model.predict(bow_X_test)

print(classification_report(y_test, y_pred))

# BoW + MLP
model = MLPClassifier(early_stopping=True, random_state=0)
model.fit(bow_X_train, y_train)
y_pred = model.predict(bow_X_test)

print(classification_report(y_test, y_pred))

"""# GloVe Embedding (not as good as BoW)"""

!pip install zeugma

from zeugma.embeddings import EmbeddingTransformer
glove = EmbeddingTransformer('glove')

glove_X_train = glove.transform(X_train.apply(lambda s: ' '.join(s)))
glove_X_test = glove.transform(X_test.apply(lambda s: ' '.join(s)))

# GloVe + LR
model  = LogisticRegression(max_iter=10000)
model.fit(glove_X_train, y_train)
y_pred = model.predict(glove_X_test)

print(classification_report(y_test, y_pred))

# GloVe + MLP
model = MLPClassifier(early_stopping=True, random_state=0)
model.fit(glove_X_train, y_train)
y_pred = model.predict(glove_X_test)

print(classification_report(y_test, y_pred))

"""# *RNN*"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import re

stemmer = SnowballStemmer('english')
text_cleaning_re = "@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+"

def preprocess(text, stem=False):
  tokens = []
  text = re.sub(text_cleaning_re, ' ', str(text).lower()).strip()
  for token in text.split():
    # if token not in stop_words:
    if stem:
      tokens.append(stemmer.stem(token))
    else:
      tokens.append(token)
  return " ".join(tokens)

X_train = X_train.apply(lambda x: preprocess(x))
X_test = X_test.apply(lambda x: preprocess(x))

from keras.preprocessing.text import Tokenizer

tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)

word_index = tokenizer.word_index
vocab_size = len(tokenizer.word_index) + 1
print("Vocabulary Size :", vocab_size)

MAX_NB_WORDS = 100000
MAX_SEQUENCE_LENGTH = 100

from keras.preprocessing.sequence import pad_sequences

X_train_pad = pad_sequences(tokenizer.texts_to_sequences(X_train), maxlen = MAX_SEQUENCE_LENGTH)
X_test_pad = pad_sequences(tokenizer.texts_to_sequences(X_test), maxlen = MAX_SEQUENCE_LENGTH)

print("Training X Shape:",X_train_pad.shape)
print("Testing X Shape:",X_test_pad.shape)

encoder = LabelEncoder()
encoder.fit(y_train)

y_train = encoder.transform(y_train)
y_test = encoder.transform(y_test)

y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)

print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

!wget http://nlp.stanford.edu/data/glove.6B.zip
!unzip glove.6B.zip

GLOVE_EMB = 'glove.6B.300d.txt'
EMBEDDING_DIM = 300
LR = 1e-3
BATCH_SIZE = 1024
EPOCHS = 50

embeddings_index = {}

# f = open(nb_path + "/" + GLOVE_EMB)
f = open(GLOVE_EMB)
for line in f:
  values = line.split()
  word = value = values[0]
  coefs = np.asarray(values[1:], dtype='float32')
  embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' %len(embeddings_index))

embedding_matrix = np.zeros((vocab_size, EMBEDDING_DIM))
for word, i in word_index.items():
  embedding_vector = embeddings_index.get(word)
  if embedding_vector is not None:
    embedding_matrix[i] = embedding_vector

embedding_layer = tf.keras.layers.Embedding(vocab_size,
                                          EMBEDDING_DIM,
                                          weights=[embedding_matrix],
                                          input_length=MAX_SEQUENCE_LENGTH,
                                          trainable=False)

from tensorflow.keras.layers import Conv1D, Bidirectional, LSTM, Dense, Input, Dropout
from tensorflow.keras.layers import SpatialDropout1D
from tensorflow.keras.callbacks import ModelCheckpoint

sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedding_sequences = embedding_layer(sequence_input)
x = Conv1D(64, 5, activation='relu')(embedding_sequences)
x = Bidirectional(LSTM(64, dropout=0.5, recurrent_dropout=0))(x)
x = Dense(64, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(32, activation='relu')(x)
outputs = Dense(1, activation='sigmoid')(x)
model = tf.keras.Model(sequence_input, outputs)

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau

model.compile(optimizer=Adam(learning_rate=LR), loss='binary_crossentropy',
              metrics=['accuracy'])
ReduceLROnPlateau = ReduceLROnPlateau(factor=0.1, min_lr = 0.01, monitor = 'val_loss', verbose = 1)

history = model.fit(X_train_pad, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS,
                    validation_data=(X_test_pad, y_test), callbacks=[ReduceLROnPlateau])

s, (at, al) = plt.subplots(2,1)
at.plot(history.history['accuracy'], c= 'b')
at.plot(history.history['val_accuracy'], c='r')
at.set_title('model accuracy')
at.set_ylabel('accuracy')
at.set_xlabel('epoch')
at.legend(['LSTM_train', 'LSTM_val'], loc='upper left')

al.plot(history.history['loss'], c='m')
al.plot(history.history['val_loss'], c='c')
al.set_title('model loss')
al.set_ylabel('loss')
al.set_xlabel('epoch')
al.legend(['train', 'val'], loc = 'upper left')

s.tight_layout()

def decode_sentiment(score):
    return 1 if score>0.5 else 0

scores = model.predict(X_test_pad, verbose=1, batch_size=10000)
y_pred = [decode_sentiment(score) for score in scores]
print(classification_report(list(y_test), y_pred))

"""# Logistic Regression for 5-score rating classification"""

import time
import seaborn as sns
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import f1_score, mean_squared_error
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Random shuffle for test set with 15000 samples
df_train_raw, df_test = train_test_split(train, test_size=15000,
                                         random_state=42, shuffle=True)

# Random shuffle for validation set with 15000 samples
df_train_raw, df_dev = train_test_split(df_train_raw, test_size=15000,
                                        random_state=42, shuffle=True)

from sklearn.utils import resample

# Create functions to extract each class of 25000 samples
def undersample(df, group_size=25000):
  dfs = []

  for label in df["Score"].value_counts().keys():
    df_group = df[df["Score"] == label]
    df_group_undersampled = resample(df_group,
                                     replace=False,
                                     n_samples=group_size,
                                     random_state=0)
    dfs.append(df_group_undersampled)

  return pd.concat(dfs).sample(frac=1, random_state=0)

# df_train = undersample(df_train_raw)

df_train = train_underSample

import matplotlib.pyplot as plt

# Plot to show training dataset distribution
score_count = df_train['Score'].value_counts()
score_count = score_count.sort_index()

fig = plt.figure(figsize=(4,3))
ax = sns.barplot(score_count.index, score_count.values)
plt.title("Review Score Distribution", fontsize=16)
plt.ylabel('Number of Reviews', fontsize=12)
plt.xlabel('Number of Scores', fontsize=12)

plt.show()
print(df_train.shape)

# Plot to show validation dataset distribution
score_count = df_dev['Score'].value_counts()
score_count = score_count.sort_index()

fig = plt.figure(figsize=(4,3))
ax = sns.barplot(score_count.index, score_count.values)
plt.title("Review Score Distribution", fontsize=16)
plt.ylabel('Number of Reviews', fontsize=12)
plt.xlabel('Number of Scores', fontsize=12)

plt.show()
print(df_dev.shape)

# Plot to show test dataset distribution
score_count = df_test['Score'].value_counts()
score_count = score_count.sort_index()

fig = plt.figure(figsize=(4,3))
ax = sns.barplot(score_count.index, score_count.values)
plt.title("Review Score Distribution", fontsize=16)
plt.ylabel('Number of Reviews', fontsize=12)
plt.xlabel('Number of Scores', fontsize=12)

plt.show()
print(df_test.shape)

from sklearn.feature_extraction.text import TfidfVectorizer

# Vectorization and preprocessing
def extract_features(df_train, df_dev, df_test):
  vectorizer = TfidfVectorizer(analyzer='word',
                               stop_words='english',
                               ngram_range=(1, 2),
                               lowercase=True,
                               min_df=5,
                               binary=False)
  X_train = vectorizer.fit_transform(df_train["Text"])
  X_dev = vectorizer.transform(df_dev["Text"])
  X_test = vectorizer.transform(df_test["Text"])
  y_train = df_train["Score"].tolist()
  y_dev = df_dev["Score"].tolist()
  y_test = df_test["Score"].tolist()

  return X_train, X_dev, X_test, y_train, y_dev, y_test

# Commented out IPython magic to ensure Python compatibility.
# %time X_train, X_dev, X_test, y_train, y_dev, y_test = extract_features(df_train, df_dev, df_test)

# Function to evaluate models
def evaluate_model_Xy(model, X, y, y_pred=None, label="Training", model_name="model"):
  if y_pred is None:
    y_pred = model.predict(X)

  print(label + ' Set')
  print("Accuracy:", accuracy_score(y, y_pred))
  print()

  print(classification_report(y, y_pred, digits=4))
  disp = plot_confusion_matrix(model, X, y,
                               cmap=plt.cm.Blues, normalize='true')
  plt.savefig(model_name + "_" + label.lower() + ".eps")
  plt.show()
  print()

# Function to evaluate models
def evaluate_model(model, model_name="model",
                   y_train_pred=None, y_dev_pred=None, y_test_pred=None):
  evaluate_model_Xy(model, X_train, y_train, label="Training", model_name=model_name)
  evaluate_model_Xy(model, X_dev, y_dev, label="Validation", model_name=model_name)
  evaluate_model_Xy(model, X_test, y_test, label="Testing", model_name=model_name)

# Instantiate the Logistic Regression model (using the default parameters)
clf_lr = LogisticRegression(penalty='l2',
                            tol=1e-4,
                            C=5.0,
                            fit_intercept=True,
                            class_weight='balanced',
                            random_state=0,
                            solver='lbfgs',
                            max_iter=100,
                            multi_class='auto',
                            verbose=1,
                            n_jobs=-1)

# Fit model to training data
clf_lr.fit(X_train, y_train)

evaluate_model(clf_lr, model_name="lr")

"""# Decison Tree for 5-score rating classification"""

import pandas as pd
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_squared_error
from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer()

import re
import numpy as np
import tensorflow as tf
import gensim.downloader as api

# # mount google drive
# from google.colab import drive
# drive.mount('/content/drive')

# data = pd.read_csv('/content/drive/My Drive/BT4222/Reviews.csv')
data = df
data = data.drop(columns=['Id'])
data = data.dropna()
data

print(data.info())

data['ProductId'].nunique()

X = data[['HelpfulnessNumerator', 'HelpfulnessDenominator']]
y = data['Score']

y.value_counts().sort_index()

from sklearn import preprocessing
le = preprocessing.LabelEncoder()

# X['ProductId'] = le.fit_transform(X['ProductId'])
dt_reg_estimator = tree.DecisionTreeRegressor()
dt_reg_estimator.fit(X, y)

y_predict = dt_reg_estimator.predict(X)
print('MSE: {}'.format(mean_squared_error(y, y_predict)))

from sklearn.model_selection import train_test_split
X_train_DT, X_test_DT, y_train_DT, y_test_DT = train_test_split(X, y, random_state=1)

# Fit a decision tree classifier with constrain on min_samples_split
dt_estimator = tree.DecisionTreeClassifier(min_samples_split=10)
dt_estimator.fit(X_train_DT, y_train_DT)

y_predict = dt_estimator.predict(X_test_DT)

report = """
The evaluation report of constrained tree is:
Confusion Matrix:
{}
Accuracy: {}
""".format(confusion_matrix(y_test_DT, y_predict),
           accuracy_score(y_test_DT, y_predict))
print(report)
