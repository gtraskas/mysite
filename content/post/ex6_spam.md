---
title: "Spam Classification"
date: 2018-02-13
tags: ["Python", "machine learning"]
draft: false
---

<script type="text/x-mathjax-config">
MathJax.Hub.Config({
  tex2jax: {
    inlineMath: [['$','$'], ['\\(','\\)']],
    displayMath: [['$$','$$'], ['\[','\]']],
    processEscapes: true,
    processEnvironments: true,
    skipTags: ['script', 'noscript', 'style', 'textarea', 'pre'],
    TeX: { equationNumbers: { autoNumber: "AMS" },
         extensions: ["AMSmath.js", "AMSsymbols.js"] }
  }
});
</script>

<script type="text/x-mathjax-config">
  MathJax.Hub.Queue(function() {
    // Fix <code> tags after MathJax finishes running. This is a
    // hack to overcome a shortcoming of Markdown. Discussion at
    // https://github.com/mojombo/jekyll/issues/199
    var all = MathJax.Hub.getAllJax(), i;
    for(i = 0; i < all.length; i += 1) {
        all[i].SourceElement().parentNode.className += ' has-jax';
    }
});
</script>

This post covers the second part of the sixth exercise from Andrew Ng's Machine Learning Course on Coursera.

***

# Spam Classification

Many email services today provide spam filters that are able to classify emails into spam and non-spam email with high accuracy. SVMs will be used to build a spam filter.

A SVM classifier will be trained  to classify whether a given email, `$x$`, is spam `$\left(y = 1 \right)$` or non-spam `$\left(y = 0 \right)$`. In particular, each email should be converted into a feature vector `$x\in\mathbb{R}^n$`.

The dataset is based on a subset of the SpamAssassin Public Corpus and only the body of the email will be used (excluding the email headers).

## Preprocessing Emails

**Sample Email:**

`Anyone knows how much it costs to host a web portal ? Well, it depends on how many visitors youre expecting. This can be anywhere from less than 10 bucks a month to a couple of $100. You should checkout http://www.rackspace.com/ or perhaps Amazon EC2 if youre running something big.. To unsubscribe yourself from this mailing list, send an email to: groupname-unsubscribe@egroups.com`

Before starting on a machine learning task, it is usually insightful to take a look at examples from the dataset. The sample email contains a URL, an email address (at the end), numbers, and dollar amounts. While many emails would contain similar types of entities (e.g., numbers, other URLs, or other email addresses), the specific entities (e.g., the specific URL or specific dollar amount) will be different in almost every email. Therefore, one method often employed in processing emails is to **"normalize"** these values, so that all URLs are treated the same, all numbers are treated the same, etc. For example, we could replace each URL in the email with the unique string "httpaddr" to indicate that a URL was present.

This has the effect of letting the spam classifier make a classification decision based on whether any URL was present, rather than whether a specific URL was present. This typically improves the performance of a spam classifier, since spammers often randomize the URLs, and thus the odds of seeing any particular URL again in a new piece of spam is very small.

In `processEmail`, the following email preprocessing and normalization steps have been implemented:

* **Lower-casing:** The entire email is converted into lower case, so that captialization is ignored (e.g., IndIcaTE is treated the same as Indicate).
* **Stripping HTML:** All HTML tags are removed from the emails. Many emails often come with HTML formatting; we remove all the HTML tags, so that only the content remains.
* **Normalizing URLs:** All URLs are replaced with the text "httpaddr".
* **Normalizing Email Addresses:** All email addresses are replaced with the text "emailaddr".
* **Normalizing Numbers:** All numbers are replaced with the text "number".
* **Normalizing Dollars:** All dollar signs ($) are replaced with the text "dollar".
* **Word Stemming:** Words are reduced to their stemmed form. For example, "discount", "discounts", "discounted" and "discounting" are all replaced with "discount". Sometimes, the Stemmer actually strips off additional characters from the end, so "include", "includes", "included", and "including" are all replaced with "includ".
* **Removal of non-words:** Non-words and punctuation have been removed. All white spaces (tabs, newlines, spaces) have all been trimmed to a single space character.

The result of these preprocessing steps looks like the following paragraph:

`anyon know how much it cost to host a web portal well it depend on how mani visitor your expect thi can be anywher from less than number buck a month to a coupl of dollarnumb you should checkout httpaddr or perhap amazon ecnumb if your run someth big to unsubscrib yourself from thi mail list send an email to emailaddr`

While preprocessing has left word fragments and non-words, this form turns out to be much easier to work with for performing feature extraction.

### Vocabulary List

After preprocessing the emails, there is a list of words for each email. The next step is to choose which words will be used in the classifier and which will be left out.

For simplicity reasons, only the most frequently occuring words as the set of words considered (the vocabulary list) have been chosen. Since words that occur rarely in the training set are only in a few emails, they might cause the model to overfit the training set. The complete vocabulary list is in the file `vocab.txt`. The vocabulary list was selected by choosing all words which occur at least a 100 times in the spam corpus, resulting in a list of 1899 words. In practice, a vocabulary list with about 10,000 to 50,000 words is often used.

Given the vocabulary list, each word can be now mapped in the preprocessed emails into a list of word indices that contains the index of the word in the vocabulary list. For example, in the sample email, the word "anyone" was first normalized to "anyon" and then mapped onto the index 86 in the vocabulary list.

The code in `processEmail` performs this mapping. In the code, a given string `str` which is a single word from the processed email is searched in the vocabulary list `vocabList`. If the word exists, the index of the word is added into the `word_indices` variable. If the word does not exist, and is therefore not in the vocabulary, the word can be skipped.


```python
# Read the txt file.
with open('emailSample1.txt', 'r') as email:
    file_contents = email.read()

file_contents
```




    "> Anyone knows how much it costs to host a web portal ?\n>\nWell, it depends on how many visitors you're expecting.\nThis can be anywhere from less than 10 bucks a month to a couple of $100. \nYou should checkout http://www.rackspace.com/ or perhaps Amazon EC2 \nif youre running something big..\n\nTo unsubscribe yourself from this mailing list, send an email to:\ngroupname-unsubscribe@egroups.com\n\n"




```python
import re
from string import punctuation
from nltk.stem.snowball import SnowballStemmer

# Create a function to read the fixed vocab list.
def getVocabList():
    """
    Reads the fixed vocabulary list in vocab.txt
    and returns a dictionary of the words in vocabList.
    """
    # Read the fixed vocabulary list.
    with open('vocab.txt', 'r') as vocab:
        
        # Store all dictionary words in dictionary vocabList.
        vocabList = {}
        for line in vocab.readlines():
            i, word = line.split()
            vocabList[word] = int(i)

    return vocabList

# Create a function to process the email contents.
def processEmail(email_contents):
    """
    Preprocesses the body of an email and returns a
    list of indices of the words contained in the email.
    Args:
        email_contents: str
    Returns:
        word_indices: list of ints
    """
    # Load Vocabulary.
    vocabList = getVocabList()

    # Init return value.
    word_indices = []
    
    # ============================ Preprocess Email ============================

    # Find the Headers ( \n\n and remove ).
    # Uncomment the following lines if you are working with raw emails with the
    # full headers.

    # hdrstart = email_contents.find("\n\n")
    # if hdrstart:
    #     email_contents = email_contents[hdrstart:]

    # Convert to lower case.
    email_contents = email_contents.lower()

    # Strip all HTML.
    # Look for any expression that starts with < and ends with > and
    # does not have any < or > in the tag and replace it with a space.
    email_contents = re.sub('<[^<>]+>', ' ', email_contents)

    # Handle Numbers.
    # Look for one or more characters between 0-9.
    email_contents = re.sub('[0-9]+', 'number', email_contents)

    # Handle URLS.
    # Look for strings starting with http:// or https://.
    email_contents = re.sub('(http|https)://[^\s]*', 'httpaddr', email_contents)

    # Handle Email Addresses.
    # Look for strings with @ in the middle.
    email_contents = re.sub('[^\s]+@[^\s]+', 'emailaddr', email_contents)

    # Handle $ sign.
    # Look for "$" and replace it with the text "dollar".
    email_contents = re.sub('[$]+', 'dollar', email_contents)


    # ============================ Tokenize Email ============================

    # Output the email to screen as well.
    print('\n==== Processed Email ====\n')

    # Process file
    l = 0
    
    # Get rid of any punctuation.
    email_contents = email_contents.translate(str.maketrans('', '', punctuation))

    # Split the email text string into individual words.
    email_contents = email_contents.split()

    for token in email_contents:

        # Remove any non alphanumeric characters.
        token = re.sub('[^a-zA-Z0-9]', '', token)
        
        # Create the stemmer.
        stemmer = SnowballStemmer("english")
        
        # Stem the word.
        token = stemmer.stem(token.strip())

        # Skip the word if it is too short
        if len(token) < 1:
           continue
        
        # Look up the word in the dictionary and add to word_indices if found.
        if token in vocabList:
            idx = vocabList[token]
            word_indices.append(idx)

        # ====================================================================

        # Print to screen, ensuring that the output lines are not too long.
        if l + len(token) + 1 > 78:
            print()
            l = 0
        print(token, end=' ')
        l = l + len(token) + 1

    # Print footer.
    print('\n\n=========================\n')

    return word_indices


# Extract features.
word_indices = processEmail(file_contents)

# Print stats.
print('Word Indices: \n')
print(word_indices)
print('\n\n')
```

    
    ==== Processed Email ====
    
    anyon know how much it cost to host a web portal well it depend on how mani 
    visitor your expect this can be anywher from less than number buck a month to 
    a coupl of dollarnumb you should checkout httpaddr or perhap amazon ecnumb if 
    your run someth big to unsubscrib yourself from this mail list send an email 
    to emailaddr 
    
    =========================
    
    Word Indices: 
    
    [86, 916, 794, 1077, 883, 370, 1699, 790, 1822, 1831, 883, 431, 1171, 794, 1002, 1895, 592, 238, 162, 89, 688, 945, 1663, 1120, 1062, 1699, 375, 1162, 479, 1893, 1510, 799, 1182, 1237, 810, 1895, 1440, 1547, 181, 1699, 1758, 1896, 688, 992, 961, 1477, 71, 530, 1699, 531]
    
    
    


## Extracting Features from Emails

The feature extraction that converts each email into a vector in `$\mathbb{R}^n$` should be implemented. For this, `$n = \text{# words in vocabulary list}$` will be used. Specifically, the feature `$x_i\in {\{0, 1\}}$` for an email corresponds to whether the i-th word in the dictionary occurs in the email. That is, `$x_i = 1$` if the i-th word is in the email and `$x_i = 0$` if the i-th word is not present in the email.

Thus, for a typical email, this feature would look like:

`$$x=\begin{bmatrix}
0 \\
\vdots\\
1\\
0\\
\vdots\\
1\\
0\\
\vdots\\
0\end{bmatrix} \in \mathbb{R}^n$$`

The code in emailFeatures generates a feature vector for an email, given the word indices. Running the code on the email sample, the feature vector will have length 1899 and 43 non-zero entries.


```python
import numpy as np

# Create a function to produce a feature vector from the word indices.
def emailFeatures(word_indices):
    """
    Takes in a word_indices vector and produces
    a feature vector from the word indices.
    Args:
        word_indices: list of ints
    Returns:
        x: binary feature vector array (n, 1)
    """
    # Total number of words in the dictionary.
    n = 1899

    # Init return value.
    x = np.zeros((n, 1))

    # Iterate over idx items in word_indices.
    for idx in word_indices:
        # Assign 1 to index idx in x.
        x[idx] = 1

    return x

# Convert each email into a vector of features in R^n.
print('Extracting features from sample email (emailSample1.txt)...\n')

# Extract features.
features = emailFeatures(word_indices)

# Print stats.
print('Length of feature vector: {:d}'.format(len(features)))
print('Number of non-zero entries: {:d}'.format(np.sum(features > 0)))
```

    Extracting features from sample email (emailSample1.txt)...
    
    Length of feature vector: 1899
    Number of non-zero entries: 43


## Training SVM for Spam Classification

Next a preprocessed training dataset will be loaded and it will be used to train a SVM classifier. `spamTrain.mat` contains 4000 training examples of spam and non-spam email, while `spamTest.mat` contains 1000 test examples. Each original email was processed using the `processEmail` and `emailFeatures` functions and converted into a vector `$x^{(i)}\in\mathbb{R}^{1899}$`.

After loading the dataset, a SVM will be trained to classify between spam `$\left(y = 1\right)$` and non-spam `$\left(y = 0\right)$` emails. Once the training completes, the classifier gets a training accuracy of about `$99.8\%$` and a test accuracy of about `$98.9\%$`.


```python
from sklearn import svm
from scipy.io import loadmat

# Load the Spam Email dataset.
email_train = loadmat('spamTrain.mat')
X = email_train["X"]
y = email_train["y"]

print('Training Linear SVM (Spam Classification)...')
print('(this may take 1 to 2 minutes) ...')

C = 0.1
y = y.ravel()
svc = svm.SVC(C, 'linear')
svc.fit(X, y)
p = svc.predict(X)

print('Training Accuracy: {0:.2f}%'.format(np.mean((p == y).astype(int)) * 100))

# Load the test dataset.
email_test = loadmat('spamTest.mat')
Xtest = email_test["Xtest"]
ytest = email_test["ytest"]

print('Evaluating the trained Linear SVM on a test set ...')

ytest = ytest.ravel()
p = svc.predict(Xtest)

print('Test Accuracy: {0:.2f}%'.format(np.mean((p == ytest).astype(int)) * 100))
```

    Training Linear SVM (Spam Classification)...
    (this may take 1 to 2 minutes) ...
    Training Accuracy: 99.83%
    Evaluating the trained Linear SVM on a test set ...
    Test Accuracy: 98.90%


## Top Predictors for Spam

To better understand how the spam classifier works, we can inspect the parameters to see which words the classifier thinks are the most predictive of spam. Next, the parameters with the largest positive values in the classifier will be found and the corresponding words will be displayed.


```python
# Get the weights.
weights = svc.coef_[0]

# Get the 15 indices that sort the most important weights.
indices = weights.argsort()[-15:]

# Reverse argsorting in descending order.
indices = (-weights).argsort()[:15]

# Return a sorted list from the dictionary.
vocabList = sorted(getVocabList())

print('Top predictors of spam: \n');
for i in indices: 
    print( '{0:10s} ({1:8f})'.format(vocabList[i], float(weights[i])))
```

    Top predictors of spam: 
    
    our        (0.500614)
    click      (0.465916)
    remov      (0.422869)
    guarante   (0.383622)
    visit      (0.367710)
    basenumb   (0.345064)
    dollar     (0.323632)
    will       (0.269724)
    price      (0.267298)
    pleas      (0.261169)
    most       (0.257298)
    nbsp       (0.253941)
    lo         (0.253467)
    ga         (0.248297)
    hour       (0.246404)


Thus, if an email contains words such as “guarantee”, “remove”, “dollar”, and “price”, it is likely to be classified as spam.
