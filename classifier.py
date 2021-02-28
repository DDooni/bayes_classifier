import re
import matplotlib.pyplot as plt
import csv
import random
import numpy as np
from scipy.interpolate import make_interp_spline, BSpline

random.seed(120)
#50, 120
def arrays_from_csv(file, ham, spam):
    with open(file, encoding='utf-8') as file:
        csv_reader = csv.reader(file, delimiter=',')
        for row in csv_reader:
            if row[0]=="ham":
                ham.append(row[1].lower())
            else:
                spam.append(row[1].lower())

def dict_from_array(arr):
    string = ""
    for i in range(len(arr)):
        string+=arr[i]
    words = words_from_string(string)
    words_freq = dict()
    for word in words:
        if words_freq.get(word)==None:
            words_freq[word]=1
        else:
            words_freq[word]+=1
    return words_freq

def words_from_string(text):
    words = re.findall(r'\b[\w\'\d]+\b', text)
    return words

def create_global_array(ham, spam):
    global_dict = dict()
    for (key, value) in ham.items():
        if global_dict.get(key)==None:
            global_dict[key]=value
        else:
            global_dict[key]+=value
    
    for (key, value) in spam.items():
        if global_dict.get(key)==None:
            global_dict[key]=value
        else:
            global_dict[key]+=value
    
    return global_dict

def dict_with_zeroes(freq_dict, global_dict):
    for (key, value) in global_dict.items():
        if freq_dict.get(key)==None:
            freq_dict[key]=0
    return freq_dict

def probs_from_freq(freq_dict, global_dict):
    words_probs = dict()
    for (key, value) in freq_dict.items():
        words_probs[key] = value / global_dict[key]
    return words_probs

def probs_normilize(probs_dict, global_dict):
    words_normilized = dict()
    for (key, value) in probs_dict.items():
        words_normilized[key]=(global_dict[key]*probs_dict[key]+0.5)/(global_dict[key]+1)
    return words_normilized

def is_spam(text, spam, ham):
    words = words_from_string(text)
    spam_prob = 0.5
    ham_prob = 0.5
    result = False
    for el in words:
        spam_prob *= spam.get(el, 1)
        ham_prob *= ham.get(el, 1)

    if spam_prob>ham_prob:
        result = True 
    return result

def classify():
    ham = []
    spam = []
    predict_percents = []
    title_amount = []

    arrays_from_csv('data.csv', ham, spam)

    random.shuffle(ham)
    random.shuffle(spam)

    while min(len(ham), len(spam))>3:
        study_amount = int(min(len(ham), len(spam))*2/3)
        
        ham_dict = dict_from_array(ham[:study_amount])
        spam_dict = dict_from_array(spam[:study_amount]) 
        global_dict = create_global_array(ham_dict, spam_dict)

        ham_dict = dict_with_zeroes(ham_dict, global_dict)
        spam_dict = dict_with_zeroes(spam_dict, global_dict)

        ham_probs = probs_from_freq(ham_dict, global_dict)
        spam_probs = probs_from_freq(spam_dict, global_dict)
        
        ham_probs = probs_normilize(ham_probs, global_dict)
        spam_probs = probs_normilize(spam_probs, global_dict)

        right = 0

        ham_t = ham[study_amount:]
        spam_t = spam[study_amount:]

        test_amount = min(len(ham_t), len(spam_t))
        
        for k in range(test_amount):
            right += int(is_spam(spam_t[k], spam_probs, ham_probs))
            right += int(not is_spam(ham_t[k], spam_probs, ham_probs))
        
        predict_percents.append((right/(test_amount*2))*100)
        title_amount.append(test_amount*2+study_amount*2)


        ham = ham[:-1]
        spam = spam[:-1]

    title_amount.reverse()
    predict_percents.reverse()

    x= np.array(title_amount)
    y= np.array(predict_percents)

    xnew = np.linspace(x.min(), x.max(), 200)
    spl = make_interp_spline(x, y, k=2)
    y_smooth = spl(xnew)

    plt.plot(xnew, y_smooth, linewidth=0.75, color='red')
    plt.title("Spam statistic")
    plt.xlabel("Titles amount")
    plt.ylabel("Accuracy, %")
    plt.show()

classify()