import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns

df = pd.read_csv('IMDB Dataset.csv', quoting=csv.QUOTE_ALL)  # Quote all fields
#QUOTE_ALL because there are unclosed quotations (EOF errors)

df['sentiment'] = df['sentiment'].map({'positive':1, 'negative': 0})

df.head()
#by default prints first 5 rows

df['sentiment'].value_counts()

X_train, X_test, y_train, y_test = train_test_split(df['review'], df['sentiment'], test_size = 0.2)

clf = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('random_forest', (RandomForestClassifier(n_estimators=50, criterion='entropy')))
])

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print('Classification Report:\n',classification_report(y_test, y_pred))

import seaborn as sns
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot = True)
plt.show()

clf = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('random_forest', (RandomForestClassifier(n_estimators=90, criterion='log_loss', random_state=20)))
])

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print('Classification Report:\n',classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot = True)
plt.show()
