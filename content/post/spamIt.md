---
title: "Spam or Ham?"
date: 2017-12-12
tags: ["nltk", "machine learning", "pandas"]
draft: false
---

Implement a spam filter in Python using the Naive Bayes algorithm to classify the emails as spam or not-spam (a.k.a. ham).

## Check Modules

Check system for the required dependencies.


```python
import sys

dependencies = ["nltk", "numpy", "pandas", "scipy", "sklearn", "pickle", "re"]

for module in dependencies:
    print("\nChecking for " + module + "...")
    try:
        # Import module from string variable:
        # https://stackoverflow.com/questions/8718885/import-module-from-string-variable
        # To import using a variable, call __import__(name)
        module_obj = __import__(module)
        # To contain the module, create a global object using globals()
        globals()[module] = module_obj
    except ImportError:
        print("Install " + module + " before continuing")
        print("In a terminal type the following commands:")
        print("python get-pip.py")
        print("pip install " + module + "\n")
        sys.exit(1)

print("\nSystem is ready!")
```

    
    Checking for nltk...
    
    Checking for numpy...
    
    Checking for pandas...
    
    Checking for scipy...
    
    Checking for sklearn...
    
    Checking for pickle...
    
    Checking for re...
    
    System is ready!


## Download Dataset

Download a set of spam and ham actual emails. Each email is a separate plain text file. Unzip the compressed tar files, read the text and load it into a Pandas Dataframe. Convert the dataframe to a Pickle object.


```python
import urllib.request
import os
import tarfile
import pickle
import pandas as pd

print("Downloading Enron emails in the Downloads folder...")

# Get the user's Downloads folder path
downloads = os.path.join(os.environ['HOME'] + "/Downloads")

url = "http://www.aueb.gr/users/ion/data/enron-spam/preprocessed/"

enron_dir = os.path.join(downloads, 'Enron emails')

enron_files = ['enron1.tar.gz', 'enron2.tar.gz', 'enron3.tar.gz',
               'enron4.tar.gz', 'enron5.tar.gz', 'enron6.tar.gz']

def download():
    """ Download Enron emails if missing. """
    
    # Create the directories.
    if not os.path.exists(enron_dir):
        os.makedirs(enron_dir)
    # Download the files that not exist.
    for file in enron_files:
        path = os.path.join(enron_dir, file)
        if not os.path.exists(path):
            urllib.request.urlretrieve(url + file, path)

def extract_emails(fname):
    """ Extract the zipped emails and load them into a pandas df.
    Args:
        fname (str): the files with tar.gz extension
    Returns:
        pandas df: a pandas dataframe of emails
    """
    
    rows = []
    tfile = tarfile.open(fname, 'r:gz')
    for member in tfile.getmembers():
        if 'ham' in member.name:
            f = tfile.extractfile(member)
            if f is not None:
                row = f.read()
                rows.append({'message': row, 'class': 'ham'})
        if 'spam' in member.name:
            f = tfile.extractfile(member)
            if f is not None:
                row = f.read()
                rows.append({'message': row, 'class': 'spam'})
    tfile.close()
    return pd.DataFrame(rows)

def populate_df_and_pickle():
    """ Populate the df with all the emails and save it to a pickle object. """
    
    if not os.path.exists(downloads + "/emails.pickle"):
        emails_df = pd.DataFrame({'message': [], 'class': []})
        for file in enron_files:
            unzipped_file = extract_emails(os.path.join(enron_dir, file))
            emails_df = emails_df.append(unzipped_file)
        emails_df.to_pickle(downloads + "/emails.pickle")

if __name__ == '__main__':
    download()
    populate_df_and_pickle()
    print("Download, unzip, and save to pickle done!")
```

    Downloading Enron emails in the Downloads folder...
    Download, unzip, and save to pickle done!



```python
with open(downloads + '/emails.pickle', 'rb') as f:
    emails_df = pickle.load(f) 

# Translate bytes objects into strings.
emails_df['message'] = emails_df['message'].apply(lambda x: x.decode('latin-1'))

# Reset pandas df index.
emails_df = emails_df.reset_index(drop=True)

# Map 'spam' to 1 and 'ham' to 0.
emails_df['class'] = emails_df['class'].map({'spam':1, 'ham':0})

print(emails_df.index)
emails_df.shape
```

    RangeIndex(start=0, stop=33716, step=1)





    (33716, 2)




```python
emails_df.iloc[25000].values
```




    array([1,
           "Subject: [ ilug - social ] prirodu requiremus social sample\r\nsocial\r\non january lst 2002 , the european countries began\r\nusing the new euro . never before have so\r\nmany countries with such powerful economies united\r\nto use a single currency . get your piece of history\r\nnow ! we would like to send you a free euro\r\nand a free report on world currency . just visit\r\nour site to request your euro and euro report :\r\nin addition to our currency report , you can receive\r\nour free investment package :\r\n* learn how $ 10 , 000 in options will leverage $ 1 , 000 , 000 in\r\neuro currency . this means even a small movement in the market\r\nhas huge profit potential . csice\r\nif you are over age 18 and have some risk capital , it ' s\r\nimportant that you find out how the euro will\r\nchange the economic world and how you can profit !\r\nplease carefully evaluate your financial position before\r\ntrading . only risk capital should be used .\r\n8 c 43 fd 25 cb 6 f 949944 eel 2 c 379 e 50028\r\nutbxcuhepuffbnkwq\r\nfull opt - out instructions on the bottom of the site\r\n- -\r\nirish linux users ' group social events : social @ linux . ie\r\nhttp : / / www . linux . ie / mailman / listinfo / social for ( un ) subscription information .\r\nlist maintainer : listmaster @ linux . ie"], dtype=object)



## Clean the Data

Remove the punctuation, any urls and numbers. Finally, convert every word to lower case.


```python
from string import punctuation
import re

def clean_email(email):
    """ Remove all punctuation, urls, numbers, and newlines.
    Convert to lower case.
    Args:
        email (unicode): the email
    Returns:
        email (unicode): only the text of the email
    """
    
    email = re.sub(r'http\S+', ' ', email)
    email = re.sub("\d+", " ", email)
    email = email.replace('\n', ' ')
    email = email.translate(str.maketrans("", "", punctuation))
    email = email.lower()
    return email

emails_df['message'] = emails_df['message'].apply(clean_email)

emails_df.iloc[25000].values
```




    array([1,
           'subject  ilug  social  prirodu requiremus social sample\r social\r on january lst    the european countries began\r using the new euro  never before have so\r many countries with such powerful economies united\r to use a single currency  get your piece of history\r now  we would like to send you a free euro\r and a free report on world currency  just visit\r our site to request your euro and euro report \r in addition to our currency report  you can receive\r our free investment package \r  learn how       in options will leverage          in\r euro currency  this means even a small movement in the market\r has huge profit potential  csice\r if you are over age   and have some risk capital  it  s\r important that you find out how the euro will\r change the economic world and how you can profit \r please carefully evaluate your financial position before\r trading  only risk capital should be used \r   c   fd   cb   f   eel   c   e  \r utbxcuhepuffbnkwq\r full opt  out instructions on the bottom of the site\r  \r irish linux users  group social events  social  linux  ie\r http    www  linux  ie  mailman  listinfo  social for  un  subscription information \r list maintainer  listmaster  linux  ie'], dtype=object)



## Prepare the Data

Split the text string into individual words and stem each word. Remove english stop words.

### Split and Stem

Split the text by white spaces and link the different forms of the same word to each other, using stemming. For example "responsiveness" and "response" have the same stem/root - "respons".

### Remove Stop Words

Some words such as “the” or “is” appear in all emails and don’t have much content to them. These words are not going to help the algorithm distinguish spam from ham. Such words are called stopwords and they can be disregarded during classification.


```python
from nltk.stem.snowball import SnowballStemmer
# nltk.download('wordnet') # uncomment to download 'wordnet'
from nltk.corpus import wordnet as wn

def preproces_text(email):
    """ Split the text string into individual words, stem each word,
    and append the stemmed word to words. Make sure there's a single
    space between each stemmed word.
    Args:
        email (unicode): the email
    Returns:
        words (unicode): the text of the email
    """
    
    words = ""
    # Create the stemmer.
    stemmer = SnowballStemmer("english")
    # Split text into words.
    email = email.split()
    for word in email:
        # Optional: remove unknown words.
        # if wn.synsets(word):
        words = words + stemmer.stem(word) + " "
    
    return words

emails_df['message'] = emails_df['message'].apply(preproces_text)

emails_df.iloc[25000].values
```




    array([1,
           'subject ilug social prirodu requiremus social sampl social on januari lst the european countri began use the new euro never befor have so mani countri with such power economi unit to use a singl currenc get your piec of histori now we would like to send you a free euro and a free report on world currenc just visit our site to request your euro and euro report in addit to our currenc report you can receiv our free invest packag learn how in option will leverag in euro currenc this mean even a small movement in the market has huge profit potenti csice if you are over age and have some risk capit it s import that you find out how the euro will chang the econom world and how you can profit pleas care evalu your financi posit befor trade onli risk capit should be use c fd cb f eel c e utbxcuhepuffbnkwq full opt out instruct on the bottom of the site irish linux user group social event social linux ie http www linux ie mailman listinfo social for un subscript inform list maintain listmast linux ie '], dtype=object)



## Machine Learning

### Vectorize Words and Split Data to Train/Test Sets

Transform the words into a tf-idf matrix using the sklearn TfIdf transformation. Then, create train/test sets with the `train_test_split` function, using `stratify` parameter. The dataset is highly unbalanced and the `stratify` parameter will make a split so that the proportion of values in the sample produced will be the same as the proportion of values provided to parameter `stratify`. For example, if variable y is 0 and 1 and there are 30% of 0's and 70% of 1's, `stratify=y` will make sure that the random split has 30% of 0's and 75% of 1's.


```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# Define the independent variables as Xs.
Xs = emails_df['message'].values

# Define the target (dependent) variable as Ys.
Ys = emails_df['class'].values

# Vectorize words - Turn the text numerical feature vectors,
# using the strategy of tokenization, counting and normalization.
vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
                                       stop_words='english')
Xs = vectorizer.fit_transform(Xs)

# Create a train/test split using 20% test size.
X_train, X_test, y_train, y_test = train_test_split(Xs,
                                                    Ys,
                                                    test_size=0.2,
                                                    shuffle=True,
                                                    random_state=0,
                                                    stratify=Ys)

feature_names = vectorizer.get_feature_names()
print("Number of different words: {0}".format(len(feature_names)))
print("Word example: {0}".format(feature_names[5369]))

# Check the split printing the shape of each set.
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)
```

    Number of different words: 119405
    Word example: arcadian
    (26972, 119405) (26972,)
    (6744, 119405) (6744,)


### Train a Classifier

Train a Naive Bayes classifier and evaluate the performance with the accuracy score.


```python
from sklearn.naive_bayes import MultinomialNB

# Create classifier.
clf = MultinomialNB()

# Fit the classifier on the training features and labels.
clf.fit(X_train, y_train)

# Make prediction - Store predictions in a list named pred.
pred = clf.predict(X_test)

# Calculate the accuracy on the test data.
print("Accuracy: {}".format(clf.score(X_test, y_test)))
```

    Accuracy: 0.9847271648873073


### Identify the Most Powerful Features

Print the 10 most important features.


```python
def get_most_important_features(vectorizer, classifier, n=None):
    feature_names = vectorizer.get_feature_names()
    top_features = sorted(zip(classifier.coef_[0], feature_names))[-n:]
    for coef, feat in top_features:
        print(coef, feat)

get_most_important_features(vectorizer, clf, 10)
```

    -7.10193040638 money
    -7.08106062291 price
    -7.07724882029 onlin
    -7.07696063312 offer
    -7.06439782381 www
    -7.04630242466 softwar
    -6.97568091654 email
    -6.94140085524 click
    -6.65836580587 com
    -6.59068342497 http


### Examples

Let's try out our classifier.


```python
email = ["Hello George, how about a game of tennis tomorrow?",
         "Hello, click here if you want to satisfy your wife tonight",
         "We offer free viagra!!! Click here now!!!",
         "Dear Sara, I prepared the annual report. Please check the attachment.",
         "Hi David, will we go for cinema tonight?",
         "Best holidays offers only here!!!"]
```


```python
examples = vectorizer.transform(email)
predictions = clf.predict(examples)
predictions
```




    array([0, 1, 1, 0, 0, 1])


