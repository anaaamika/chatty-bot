import nltk
import numpy as np
import random
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import tkinter
from tkinter import *

f = open('chatdata.txt', 'r', errors='ignore')

raw = f.read()

raw = raw.lower()

# nltk.download('punkt')
# nltk.download('wordnet')

sent_tokens = nltk.sent_tokenize(raw)
word_tokens = nltk.word_tokenize(raw)

lemmer = nltk.stem.WordNetLemmatizer()

def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]


remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)


def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))


GREETING_INPUTS = ("hello", "hi", "greetings", "sup", "what's up", "hey")
GREETING_RESPONSES = ['hi', 'hey', '*nods*', 'hi there', 'hello', 'I am glad! You are talking to me']


def greeting(sentence):
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)


def response(user_response):
    robo_response = ''
    sent_tokens.append(user_response)

    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = TfidfVec.fit_transform(sent_tokens)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx = vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]

    if req_tfidf == 0:
        robo_response = robo_response + "I am sorry! I don't understand you"
        return robo_response
    else:
        robo_response = robo_response + sent_tokens[idx]
        return robo_response


# building GUI using tkinter
def send():
    msg = EntryBox.get('1.0', 'end-1c').strip()
    EntryBox.delete('0.0', END)

    if msg != '':
        ChatLog.config(state=NORMAL)
        ChatLog.insert(END, 'You: ' + msg + '\n\n')
        # ChatLog.config(foreground='#442265', font=('Verdana', 12))

        if msg != "bye":
            if msg == 'thanks' or msg == 'thank you':
                res = "You are welcome..."
            else:
                if greeting(msg) != None:
                    res = greeting(msg)
                else:
                    res = response(msg)
                    sent_tokens.remove(msg)
        else:
            res = "Bye! Take care..."
        ChatLog.insert(END, 'Chatty: ' + res + '\n\n')

        ChatLog.config(state=DISABLED)
        ChatLog.yview(END)


base = Tk()
base.title('Hello!')
base.geometry('400x500')
base.resizable(width=FALSE, height=FALSE)

# Create chat window
ChatLog = Text(base, bd=0, bg='white', height='8', width='50', foreground='#442265', font=('Verdana', 12))
ChatLog.config(wrap=WORD)
ChatLog.insert(END,
               'Chatty: Hi there! My name is Chatty. I can answer your questions about life, the universe, and everything.'
               + '\n' + 'If you want to exit, type Bye!'
               + '\n\n' + 'Please enter a topic...' + '\n\n')
ChatLog.config(state=DISABLED)

# Bind scrollbar to chat window
scrollbar = Scrollbar(base, command=ChatLog.yview, cursor='heart')
ChatLog['yscrollcommand'] = scrollbar.set

# Create button to send message
SendButton = Button(base, font=('Verdana', 12, 'bold'), text='Send', width='10', height='5', bd=0,
                    bg='#32de97', fg='#ffffff', command=send)

# Create user input box
EntryBox = Text(base, bd=0, bg='white', width='29', height='5', foreground='#442265', font=('Verdana', 12))
# EntryBox.bind('<Return>', send)

# Place all components on screen
scrollbar.place(x=376, y=6, height=431)
ChatLog.place(x=6, y=6, height=431, width=370)
EntryBox.place(x=6, y=446, height=45, width=265)
SendButton.place(x=276, y=446, height=45)

base.mainloop()
