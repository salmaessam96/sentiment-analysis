
import pandas as pd
import re
import csv
import string
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-@.&+]|[!*(),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
edited = []
tweet = []
with open('train.csv', 'r',encoding="utf8") as file:
    data = csv.reader(file)
    for row in data:
        # Replace URLs with an empty string
        row[2] = url_pattern.sub('', row[2])
        # Remove punctuation
        row[2] = re.sub('['+string.punctuation+']', '', row[2])
        # replaces all non-ASCII characters to space
        row[2] = re.sub(r'[^\x00-\x7F]+', ' ', row[2])
        #replace numbers with spaces
        row[2] = re.sub(r'\d+', ' ', row[2])
        #remove white space
        row[2] = re.sub(' +', ' ', row[2])
        #lowercase
        row[2] = row[2].lower()
        # removing words(our,the,under ,like, through..........)
        row[2] = row[2].split()
        row[2] = [word for word in row[2] if word not in stopwords.words('english')]
        row[2] = ' '.join(row[2])
        #print(row[2])
        edited.append([row[0],row[1],row[2]])
        tweet.append(row[2])

filename = "edited.csv"

# writing to csv file
with open(filename, 'w',encoding="utf8", newline='') as csvfile:
    # creating a csv writer object
    csvwriter = csv.writer(csvfile)

    # writing the data rows
    for row in edited:
        csvwriter.writerow(row)

data_clean = pd.read_csv('edited.csv')
Y = data_clean['label']

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=6926)
X = tweet[1:]
X = cv.fit_transform(X).toarray()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=42)

from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB()
clf. fit(X_train, y_train)

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
y_pred = clf.predict(X_test)

print(confusion_matrix(y_test, y_pred))
print(accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

edited_test=[]
with open('test.csv', 'r',encoding="utf8") as file:
    data = csv.reader(file)
    for row in data:
        # Replace URLs with an empty string
        row[1] = url_pattern.sub('', row[1])
        # Remove punctuation without hashtags
        row[1] = re.sub('['+string.punctuation+']', '', row[1])
        # replaces all non-ASCII characters to space
        row[1] = re.sub(r'[^\x00-\x7F]+', ' ', row[1])
        #replace numbers with spaces
        row[1] = re.sub(r'\d+', ' ', row[1])
        #remove white space
        row[1] = re.sub(' +', ' ', row[1])
        #lowercase
        row[1] = row[1].lower()
        # removing words(our,the,under ,like, through..........)
        row[1] = row[1].split()
        row[1] = [word for word in row[1] if word not in stopwords.words('english')]
        row[1] = ' '.join(row[1])
        #print(row[1])
        edited_test.append([row[0],row[1]])
        #print(edited,'\n')

filename = "edited_test.csv"

# writing to csv file
with open(filename, 'w',encoding="utf8", newline='') as csvfile:
    # creating a csv writer object
    csvwriter = csv.writer(csvfile)

    # writing the data rows
    for row in edited_test:
        csvwriter.writerow(row)

data_train = pd.read_csv('edited_test.csv')
y = data_train['tweet']
#print(len(y))
test = cv.transform(y)
id = data_train['id']
test_pred = clf.predict(test)

with open("sample_submission_LnhVWA4.csv", 'w',encoding="utf8", newline='') as csvfile:
    # creating a csv writer object
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(['id', 'label'])
    for i in range(len(id)):
        csvwriter.writerow([id[i], test_pred[i]])