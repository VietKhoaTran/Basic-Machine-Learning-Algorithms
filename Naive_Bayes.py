import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
import sklearn.datasets
from sklearn.metrics import accuracy_score
#-----------------------Learning------
emails = [
    ('free money', 'spam'),
    ('free offer', 'spam'),
    ('buy now', 'not spam'),  
    ('hello friends', 'not spam'),
    ('free money offer', 'spam'),
    ('hello', 'not spam'),
]

x = [i[0].split() for i in emails]
y = [i[1] for i in emails]

total_emails = len(emails)
class_counts = {label: y.count(label) for label in set(y)}
prior_props = {label: count/total_emails for label, count in class_counts.items()}

all_words = set(word for word2 in x for word in word2)
word_counts = {label:{word: 1 for word in all_words} for label in set(y)}
for email, label in zip(x, y):
    for word in email:
        word_counts[label][word] +=1

for label in word_counts:
    total_words = sum(word_counts[label].values())
    for word in word_counts[label]:
        word_counts[label][word] /= total_words
#-----------------------------------Testing------------
def test(new_email):
    new_email_words = new_email.split()
    probabilities = {}

    for label in word_counts:
        likelihood = 1
        for word in new_email_words:
            likelihood *= word_counts[label].get(word, 1e-10)
        probabilities[label] = prior_props[label] * likelihood
    return max(probabilities, key=probabilities.get)

new_email = "free offer and money"
predicted_label = test(new_email)
print(f"Predicted Label for '{new_email}': {predicted_label}")