# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np
import re
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
from textblob import TextBlob
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

df = pd.read_csv('https://raw.githubusercontent.com/amankharwal/Website-data/master/vaccination_tweets.csv')

df.head()

df.info()

df.isnull().sum()

df.columns

text_df = df.drop(['id', 'user_name', 'user_location', 'user_description', 'user_created',
       'user_followers', 'user_friends', 'user_favourites', 'user_verified',
       'date', 'hashtags', 'source', 'retweets', 'favorites',
       'is_retweet'], axis=1)
text_df.head()

print(text_df['text'].iloc[0],"\n")
print(text_df['text'].iloc[1],"\n")
print(text_df['text'].iloc[2],"\n")
print(text_df['text'].iloc[3],"\n")
print(text_df['text'].iloc[4],"\n")

text_df.info()

def data_processing(text):
    text = text.lower()
    text = re.sub(r"https\S+|www\S+https\S+", '',text, flags=re.MULTILINE)
    text = re.sub(r'\@w+|\#','',text)
    text = re.sub(r'[^\w\s]','',text)
    text_tokens = word_tokenize(text)
    filtered_text = [w for w in text_tokens if not w in stop_words]
    return " ".join(filtered_text)

text_df = text_df['text'].apply(data_processing)

text_df = text_df.drop_duplicates('text')

stemmer = PorterStemmer()
def stemming(data):
    text = [stemmer.stem(word) for word in data]
    return data

text_df['text'] = text_df['text'].apply(lambda x: stemming(x))

text_df.head()

print(text_df['text'].iloc[0],"\n")
print(text_df['text'].iloc[1],"\n")
print(text_df['text'].iloc[2],"\n")
print(text_df['text'].iloc[3],"\n")
print(text_df['text'].iloc[4],"\n")

text_df.info()

def polarity(text):
    return TextBlob(text).sentiment.polarity

text_df['polarity'] = text_df['text'].apply(polarity)

text_df.head(10)

def sentiment(label):
    if label <0:
        return "Negative"
    elif label ==0:
        return "Neutral"
    elif label>0:
        return "Positive"

text_df['sentiment'] = text_df['polarity'].apply(sentiment)

text_df.head()

fig = plt.figure(figsize=(5, 5))
sns.countplot(y='sentiment', data=text_df)  # Use 'y' to create a horizontal bar chart

plt.title('Distribution of sentiments')
plt.xlabel('Count')
plt.ylabel('Sentiment')

plt.show()

import matplotlib.pyplot as plt

# Data for the donut chart
labels = text_df['sentiment'].value_counts().index
sizes = text_df['sentiment'].value_counts().values
colors = ["lightblue", "lightgreen", "lightcoral"]
explode = (0.1, 0.1, 0.1)

# Create a donut chart
fig, ax = plt.subplots()
ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
       startangle=90, pctdistance=0.85, explode=explode)

# Draw a circle in the center to create the donut hole
center_circle = plt.Circle((0, 0), 0.70, fc='white')
fig.gca().add_artist(center_circle)

# Equal aspect ratio ensures that pie is drawn as a circle
ax.axis('equal')
plt.title('Distribution of sentiments (Donut Chart)', fontsize=14)

# Show the donut chart
plt.show()

pos_tweets = text_df[text_df.sentiment == 'Positive']
pos_tweets = pos_tweets.sort_values(['polarity'], ascending= False)
pos_tweets.head()

from wordcloud import WordCloud
import matplotlib.pyplot as plt

text = ' '.join([word for word in pos_tweets['text']])

# Customize the WordCloud parameters
wordcloud = WordCloud(
    background_color='white',  # Background color
    max_words=500,              # Maximum number of words to display
    width=800,                  # Width of the image
    height=400,                 # Height of the image
    colormap='viridis',         # Color map (you can choose from available colormaps)
    contour_color='steelblue',  # Contour color
    contour_width=2,            # Contour line width
).generate(text)

# Create a figure and axis with custom size and appearance
plt.figure(figsize=(12, 6), facecolor='white')
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")  # Turn off axis
plt.title('Most Frequent Words in Positive Tweets', fontsize=16, color='steelblue')
plt.show()

neg_tweets = text_df[text_df.sentiment == 'Negative']
neg_tweets = neg_tweets.sort_values(['polarity'], ascending= False)
neg_tweets.head()

from wordcloud import WordCloud
import matplotlib.pyplot as plt

text = ' '.join([word for word in neg_tweets['text']])

# Customize the WordCloud parameters
wordcloud = WordCloud(
    background_color='white',  # Background color
    max_words=500,              # Maximum number of words to display
    width=800,                  # Width of the image
    height=400,                 # Height of the image
    colormap='viridis',         # Color map (you can choose from available colormaps)
    contour_color='steelblue',  # Contour color
    contour_width=2,            # Contour line width
).generate(text)

# Create a figure and axis with custom size and appearance
plt.figure(figsize=(12, 6), facecolor='white')
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")  # Turn off axis
plt.title('Most Frequent Words in  Negitive Tweets', fontsize=16, color='steelblue')
plt.show()

neutral_tweets = text_df[text_df.sentiment == 'Neutral']
neutral_tweets = neutral_tweets.sort_values(['polarity'], ascending= False)
neutral_tweets.head()

from wordcloud import WordCloud
import matplotlib.pyplot as plt

text = ' '.join([word for word in neutral_tweets['text']])

# Customize the WordCloud parameters
wordcloud = WordCloud(
    background_color='white',  # Background color
    max_words=500,              # Maximum number of words to display
    width=800,                  # Width of the image
    height=400,                 # Height of the image
    colormap='viridis',         # Color map (you can choose from available colormaps)
    contour_color='steelblue',  # Contour color
    contour_width=2,            # Contour line width
).generate(text)

# Create a figure and axis with custom size and appearance
plt.figure(figsize=(12, 6), facecolor='white')
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")  # Turn off axis
plt.title('Most Frequent Words in neutral Tweets', fontsize=16, color='steelblue')
plt.show()

vect = CountVectorizer(ngram_range=(1,2)).fit(text_df['text'])

X = text_df['text']
Y = text_df['sentiment']
X = vect.transform(X)

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

print("Size of x_train:", (x_train.shape))
print("Size of y_train:", (y_train.shape))
print("Size of x_test:", (x_test.shape))
print("Size of y_test:", (y_test.shape))

import warnings
warnings.filterwarnings('ignore')

logreg = LogisticRegression()
logreg.fit(x_train, y_train)
logreg_pred = logreg.predict(x_test)
logreg_acc = accuracy_score(logreg_pred, y_test)
print("Test accuracy: {:.2f}%".format(logreg_acc*100))

print(confusion_matrix(y_test, logreg_pred))
print("\n")
print(classification_report(y_test, logreg_pred))

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Assuming you have already computed the confusion matrix and have it in the 'cm' variable

# Create a figure and customize the style
plt.figure(figsize=(8, 6), dpi=80)
plt.style.use('seaborn')  # Change the style to 'seaborn' (you can choose from other styles)

# Create a ConfusionMatrixDisplay object
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=logreg.classes_)

# Plot the confusion matrix with desired settings
disp.plot(cmap='Blues', values_format='.4g')  # Change the colormap and format of values

# Customize the appearance of the plot
plt.title('Confusion Matrix', fontsize=14)
plt.xlabel('Predicted', fontsize=12)
plt.ylabel('Actual', fontsize=12)

# Save the figure in a specific format (e.g., PNG)
plt.savefig('confusion_matrix.png', format='png', bbox_inches='tight')

# Show the plot
plt.show()

from sklearn.model_selection import GridSearchCV

param_grid={'C':[0.001, 0.01, 0.1, 1, 10]}
grid = GridSearchCV(LogisticRegression(), param_grid)
grid.fit(x_train, y_train)

print("Best parameters:", grid.best_params_)

y_pred = grid.predict(x_test)

logreg_acc = accuracy_score(y_pred, y_test)
print("Test accuracy: {:.2f}%".format(logreg_acc*100))

print(confusion_matrix(y_test, y_pred))
print("\n")
print(classification_report(y_test, y_pred))

from sklearn.svm import LinearSVC

SVCmodel = LinearSVC()
SVCmodel.fit(x_train, y_train)

svc_pred = SVCmodel.predict(x_test)
svc_acc = accuracy_score(svc_pred, y_test)
print("test accuracy: {:.2f}%".format(svc_acc*100))

print(confusion_matrix(y_test, svc_pred))
print("\n")
print(classification_report(y_test, svc_pred))

grid = {
    'C':[0.01, 0.1, 1, 10],
    'kernel':["linear","poly","rbf","sigmoid"],
    'degree':[1,3,5,7],
    'gamma':[0.01,1]
}
grid = GridSearchCV(SVCmodel, param_grid)
grid.fit(x_train, y_train)

y_pred = grid.predict(x_test)

logreg_acc = accuracy_score(y_pred, y_test)
print("Test accuracy: {:.2f}%".format(logreg_acc*100))

print(confusion_matrix(y_test, y_pred))
print("\n")
print(classification_report(y_test, y_pred))

import tweepy #to access the twitter api
import pandas as pd #for basic data operations

