#%%
import numpy as np
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
#%% [markdown]
# Let's make sure our processing of these files worked correctly
#%%
df = pd.read_csv('Data/movie_data.csv', encoding='utf-8')
print(df.head())
print(df.shape)
#%% [markdown]
## Cleaning the text
# We're going to have to clean any scraps of html and random punctuation from
# our movie reviews in order to get good results from our model. We will do this
# with regular expressions.
#%%


def preprocessor(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)',
                           text)
    text = re.sub('(?::|;|=)(?:-)?(?:\)|\(|D|P)', '', text)
    text = (re.sub('[\W]+', ' ', text.lower()) +
            ' '.join(emoticons).replace('-', ''))
    return text
test_string = '<a>This is a test! :) :-D</a>'
valid_string = 'this is a test :) :D'
print(preprocessor(test_string))
print('Preprocessor: %r' % (valid_string == preprocessor(test_string)))
df['review'] = df['review'].apply(preprocessor)
#%% [markdown]
# Our preprocessor works, let's make a tokenizer that uses porter stemming to
# sort of find the root words. We'll also make one with no stemming to see if
# stemming actually helps
#%%
from nltk.stem.porter import PorterStemmer

porter = PorterStemmer()

def tokenizer_porter(text):
    return [porter.stem(word) for word in text.split()]

def tokenizer(text):
    return text.split()

test_string = '<a>My dog Goji is a cutie little lover :) and good dogger!'
processed = preprocessor(test_string)
tokenizer_porter(processed)
#%% [markdown]
# Unclear if this is good or not but whatever let's move on and look briefly at
# stopwords
#%%
from nltk.corpus import stopwords

stop = stopwords.words('english')
[word for word in tokenizer_porter(processed) if word not in stop]

#%% [markdown]
# Ok, let's get into it. Make training and test sets
#%%
X_train = df.loc[:25000, 'review'].values
y_train = df.loc[:25000, 'sentiment'].values
X_test = df.loc[25000:, 'review'].values
y_test = df.loc[25000:, 'sentiment'].values
#%% [markdown]
# We'll use logistic regression as in the book. We might also try Naive Bayes if
# we are feeling spicy.
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

tfidf = TfidfVectorizer(strip_accents=False,
                        lowercase=False,
                        preprocessor=None)

param_grid = [{'vect__ngram_range' : [(1, 1)],
               'vect__stop_words' : [stop, None],
               'vect__tokenizer' : [tokenizer],
               'clf__penalty' : ['l1', 'l2'],
               'clf__C' : [1.0, 10.0, 100.0]},
              {'vect__ngram_range': [(1, 1)],
               'vect__stop_words': [stop, None],
               'vect__tokenizer': [tokenizer],
               'clf__penalty': ['l1', 'l2'],
               'clf__C': [1.0, 10.0, 100.0],
               'vect__use_idf' : [None],
               'vect__norm' : [None]}]

pipe_lr = Pipeline([('vect', tfidf),
                   ('clf', LogisticRegression(random_state=666))])
                   
gs_lr_tfidf = GridSearchCV(estimator=pipe_lr,
                           param_grid=param_grid,
                           scoring='accuracy',
                           cv=5,
                           verbose=100,
                           n_jobs=-1)

#%%
gs_lr_tfidf.fit(X_train, y_train)

#%%
clf = gs_lr_tfidf.best_estimator_
print('CV Accuracy: %.3f' % (gs_lr_tfidf.best_score_))
print(('Test Accuracy: %.3f' % clf.score(X_test, y_test)))

#%% [markdown]
## Out-of-core learning
# Now let's write this thing with out-of-core learning. First we need to
# (re)write some functions starting with our tokenizer
#%%
def tokenizer_stopper(text):
    text = re.sub(r'<[^>]*>', '', text)
    emoticons = re.findall(r'(?::|;|=)(?:-)?(?:\)|\(|D|P)',
                           text)
    text = re.sub(r'(?::|;|=)(?:-)?(?:\)|\(|D|P)', '', text)
    text = (re.sub(r'[\W]+', ' ', text.lower()) +
            ' '.join(emoticons).replace('-', ''))
    tokenized = [w for w in text.split() if w not in stop]
    return tokenized

def stream_docs(path):
    with open(path, 'r', encoding='utf-8') as csv:
        next(csv)
        for line in csv:
            text, label = line[:-3], int(line[-2])
            yield text, label

def get_minibatch(doc_stream, size):
    docs, y = [], []
    try:
        for _ in range(size):
            text, label = next(doc_stream)
            docs.append(text)
            y.append(label)
    except StopIteration:
        return None, None
    return docs, y

#%% [markdown]
# Now we will make the classifier. We can't just do LogisticRegression because
# we need to utilize stochastic gradient descent so we will use SGDClassifier
# with the 'log' setting (which is logistic regression). We will also need to
# use a different vectorizer.
#%%
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
vect = HashingVectorizer(decode_error='ignore',
                         n_features=2**21,
                         preprocessor=None,
                         tokenizer=tokenizer_stopper)
clf = SGDClassifier(loss='log', random_state=1, max_iter=1)
doc_stream = stream_docs(path='data/movie_data.csv')

#%% [markdown]
# Now we will train this bad boy
#%%
import pyprind
pbar = pyprind.ProgBar(45)
classes = np.array([0, 1])
for _ in range(45):
    X_train, y_train = get_minibatch(doc_stream, size=1000)
    if not X_train:
        break
    X_train = vect.transform(X_train)
    clf.partial_fit(X_train, y_train, classes=classes)
    pbar.update()

X_test, y_test = get_minibatch(doc_stream, size=5000)
X_test = vect.transform(X_test)
print('Accuracy: %.3f' % clf.score(X_test, y_test))
#%%
clf.partial_fit(X_test, y_test)
#%% [markdown]
## Serializing our estimator
# we will now save our fitted model using the pickle module of python
#%%
import pickle
import os
dest = os.path.join('movieclassifier', 'pkl_objects')
if not os.path.exists(dest):
    os.makedirs(dest)

pickle.dump(stop, open(os.path.join(dest, 'stopwords.pkl'), 'wb'), protocol=4)
pickle.dump(clf, open(os.path.join(dest, 'classifier.pkl'), 'wb'), protocol=4)

#%%
