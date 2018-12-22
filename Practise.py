from tkinter import *
from tkinter import ttk, StringVar
from tkinter.ttk import Combobox
from tkinter import filedialog
from tkinter import messagebox
import sqlite3
from tkinter import ttk
from tkinter import filedialog
import  random
# Load libraries
import pandas
from matplotlib.artist import get
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import Normalizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import euclidean_distances
from sklearn import tree
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import io
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer
import numpy as np
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
import gensim
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from nltk.corpus import stopwords
import numpy as np
import numpy.linalg as LA
import sys
from pydoc import help
import os
from tkinter import messagebox



list=[]
list2=[]

def open_file():
    lbl4 = Label(frame4, text='Add CV', font=('arial', 10, 'bold'))
    lbl4.grid(row=3, column=2)
    raw_documents = filedialog.askopenfile(initialdir="/", title="Choose a file",
                                    filetypes=(("text files", ".txt"), ("all files", "*.*")))

    lbl4.configure(text=raw_documents.name)

    for w in raw_documents:
        list.append(w)

def analyze():
    rltlbl=Label(frame4, padx=15, pady=15, width=20, font=("bold", 10), bg="white")
    rltlbl.grid(column=2, row=5, pady=10)
    rltlbl.configure(text=["Number of document: ",len(list)])
    rltlbl2 = Label(frame4, padx=15, pady=15, font=("bold", 10), bg="white")
    rltlbl2.grid(column=2, row=6)

    gen_docs = [[w.lower() for w in word_tokenize(text)]
                for text in list]
    #print(gen_docs)

    dictionary = gensim.corpora.Dictionary(gen_docs)
    corpus = [dictionary.doc2bow(gen_doc) for gen_doc in gen_docs]
    #print(corpus)
    tf_idf = gensim.models.TfidfModel(corpus)
    # print(tf_idf)
    s = 0
    for i in corpus:
        s += len(i)
    # print(s)
    sims = gensim.similarities.Similarity('C:\Similarity\sims', tf_idf[corpus],

                                        num_features=len(dictionary))

    query=tb.get("1.0", "end-1c")
    query_doc = [w.lower() for w in word_tokenize(query)]
    #print(query_doc)
    query_doc_bow = dictionary.doc2bow(query_doc)
    #print(query_doc_bow)
    query_doc_tf_idf = tf_idf[query_doc_bow]
    #sims[query_doc_tf_idf]
    # print(query_doc_tf_idf)
    rlt=('Query result:', sims[query_doc_tf_idf])
    rltlbl2.configure(text=rlt)


def combo_input():
    db = sqlite3.connect('JobsDetails.db')
    cursor = db.execute('SELECT JobID from Jobs')
    result = []
    for row in cursor.fetchall():
        result.append(row[0])
    return result

def view_data(self):
    global tb
    lbl3 = Label(frame4, text='Requirments', font=('arial', 10, 'bold'))
    lbl3.grid(row=1, column=0)
    tb = Text(frame4, padx=15, pady=15, width=25, height=10, font=("bold", 10), bg="#00ffbf")
    tb.grid(row=1, column=2)

    lbl3 = Label(frame4, text='Add CV',font=('arial', 10, 'bold'))
    lbl3.grid(row=2, column=0)
    cvbtn = Button(frame4, text="Open file", width=22, font=("bold", 12), command=open_file)
    cvbtn.grid(column=2, row=2,pady=10 )

    cvbtn = Button(frame4, text="Analyze", width=22, font=("bold", 12), command=analyze)
    cvbtn.grid(column=2, row=4, pady=10)

    db = sqlite3.connect('JobsDetails.db')
    cursor = db.cursor()
    # cursor.execute("SELECT * FROM Jobs")
    find = ("SELECT Qualification,Experience,Skills from Jobs WHERE JobID=?")
    cursor.execute(find, [(combo.get())])
    rows = cursor.fetchall()
    #for row in rows:
    tb.insert("1.0", rows)

def error():
    messagebox.showerror("Error", "Please insert data!")

def addjob():
    jobid = JobId.get()
    deg = Designation.get()
    qln = Qualification.get()
    exp = Experience.get()
    skills = textbox.get("1.0", "end-1c")

    db = sqlite3.connect('JobsDetails.db')
    with db:
        cursor = db.cursor()
    cursor.execute(
        'CREATE TABLE IF NOT EXISTS Jobs (JobID TEXT, Designation TEXT,Qualification TEXT, Experience TEXT,Skills TEXT)')
    find_er = ("SELECT * from Jobs WHERE JobID=?")
    cursor.execute(find_er, [(JobId.get())])

    if jobid == "":
        error()
    elif qln == "":
        error()
    elif exp == "":
        error()
    elif skills == "":
        error()
    elif cursor.fetchall():
        messagebox.showerror("Error", "ID Already taken")
    else:
        messagebox.showinfo("Success", "New Jobs added!")
        # save in database
        cursor.execute('INSERT INTO Jobs (JobID,Designation,Qualification,Experience,Skills) VALUES(?,?,?,?,?)',
                       (jobid, deg, qln, exp, skills,))

        db.commit()


def reset():
    JobId.set("")
    Designation.set("")
    Qualification.set("")
    Experience.set("")
    textbox.delete("1.0", "end")


def exit():
    exit = messagebox.askyesno("CV Shortlisting Tool", "Are You Syre want to Exit?")
    if exit > 0:
        win1.destroy()
        return

def win2():
    global win2
    win2=Toplevel(win1)
    win2.geometry("1350x700+0+0")
    win2.title("Shortlisting Tool")
    global combo
    global frame4
    frame3 = Frame(win2)
    frame3.pack()
    frame4 = Frame(frame3, bd=5, height=500, width=500, relief='ridge')
    frame4.grid(row=1, column=0)

    lbltitle = Label(frame3, text="Analysis CV", font=('arial', 25, 'bold'), bd=20)
    lbltitle.grid(row=0, column=0, columnspan=2, pady=10)

    lbltitle = Label(frame4, text="Select Job ID", font=('arial', 10, 'bold'), bd=20)
    lbltitle.grid(row=0, column=0, columnspan=2, pady=10)

    combo = Combobox(frame4, width=20, font=("bold", 12), state='readonly')
    combo.set("Select")
    combo['value'] = combo_input()
    combo.bind("<<ComboboxSelected>>", view_data)
    combo.grid(row=0, column=2, padx=100)

def main_window():
    global win1
    win1=Tk()
    win1.geometry("1350x700+0+0")
    win1.title("Shortlisting Tool")

    global JobId
    global Designation
    global Qualification
    global Experience
    global textbox
    JobId = StringVar()
    Designation = StringVar()
    Qualification = StringVar()
    Experience = StringVar()

    frame = Frame(win1)
    frame.pack()
    frame1 = Frame(frame, bd=5, height=500, width=500, relief='ridge')
    frame1.grid(row=1, column=0)
    lbltitle = Label(frame, text="Add Jobs Details", font=('arial', 25, 'bold'), bd=20)
    lbltitle.grid(row=0, column=0, columnspan=2, pady=10)

    lbl1 = Label(frame1, text='Job ID', font=('arial', 10, 'bold'), bd=20)
    lbl1.grid(row=0, column=0)
    ent1 = Entry(frame1, width=30, bd=3, textvar=JobId)
    ent1.grid(row=0, column=2, padx=100)

    lbl2 = Label(frame1, text='Designation', font=('arial', 10, 'bold'), bd=20)
    lbl2.grid(row=1, column=0, )
    ent2 = Entry(frame1, width=30, bd=3, textvar=Designation)
    ent2.grid(row=1, column=2)

    lbl2 = Label(frame1, text='Qualification', font=('arial', 10, 'bold'), bd=20)
    lbl2.grid(row=2, column=0, pady=10)
    ent2 = Entry(frame1, width=30, bd=3, textvar=Qualification)
    ent2.grid(row=2, column=2)

    lbl2 = Label(frame1, text='Experience', font=('arial', 10, 'bold'), bd=20)
    lbl2.grid(row=3, column=0)
    ent2 = Entry(frame1, width=30, textvar=Experience, bd=3)
    ent2.grid(row=3, column=2)

    lbl2 = Label(frame1, text='Key Skills', font=('arial', 10, 'bold'), bd=20)
    lbl2.grid(row=4, column=0)
    textbox = Text(frame1, undo=True, width=23, height=7, bd=3)
    textbox.grid(row=4, column=2, pady=10)

    frame2 = Frame(frame, width=800, height=50, bd=5, relief='ridge')
    frame2.grid(row=2, column=0)

    addbtn = Button(frame2, text="Submit", width=12, font=('arial', 12, 'bold'), command=addjob)
    addbtn.grid(row=0, column=2)

    resetbtn = Button(frame2, text="Reset", width=11, font=('arial', 12, 'bold'), command=reset)
    resetbtn.grid(row=0, column=3)

    analysebtn = Button(frame2, text="Analysis CV", width=12, font=('arial', 12, 'bold'), command=win2)
    analysebtn.grid(row=0, column=4)

    exitbtn = Button(frame2, text="Exit", width=12, font=('arial', 12, 'bold'), command=exit)
    exitbtn.grid(row=0, column=5)

    win1.mainloop()

main_window()


"""
window = Tk()
window.geometry("700x600+250+0")
window.title("Cv Shortlisting Tool")

menu = Menu(window, bg='red')
new_item = Menu(menu)
menu.add_cascade(label='File', menu=new_item)
menu.add_cascade(label='Edit', menu=new_item)
window.config(menu=menu)
topFrame=Frame(window)
topFrame.pack()



stop_words = set(stopwords.words('english'))
list=[]
list2=[]
def open_file():
    raw_documents = filedialog.askopenfile(initialdir="/", title="Choose a file",
                                    filetypes=(("text files", ".txt"), ("all files", "*.*")))
    fname=raw_documents.name
    list2.append(fname)
    lbl4.configure(text=list2)

    for w in raw_documents:
        list.append(w)

def analyze():
    print("Number of documents:", len(list))
    #print(list)

    gen_docs = [[w.lower() for w in word_tokenize(text)]
                for text in list]
    #print(gen_docs)

    dictionary = gensim.corpora.Dictionary(gen_docs)
    corpus = [dictionary.doc2bow(gen_doc) for gen_doc in gen_docs]
    #print(corpus)

    tf_idf = gensim.models.TfidfModel(corpus)
    # print(tf_idf)
    s = 0
    for i in corpus:
        s += len(i)
    #print(s)
    sims = gensim.similarities.Similarity('C:\Similarity\sims', tf_idf[corpus],
                                          num_features=len(dictionary))
    query=textbox.get("1.0","end-1c")
    query_doc = [w.lower() for w in word_tokenize(query)]
    #print(query_doc)
    query_doc_bow = dictionary.doc2bow(query_doc)
    #print(query_doc_bow)
    query_doc_tf_idf = tf_idf[query_doc_bow]
    # sims[query_doc_tf_idf]
    #print(query_doc_tf_idf)
    rel=('Query result:', sims[query_doc_tf_idf])
    lbl5 = Label(topFrame,font=("bold", 10), bg="#00ffbf")
    lbl5.grid(column=2, row=8)
    lbl5.configure(text=rel)

lbl1 = Label(topFrame, text='Job ID', padx=15, pady=15, width=20,font=("bold", 12) )
lbl1.grid(row=1,column=1)
ent1=Entry(topFrame, width=40)
ent1.grid(column=2, row=1)

lbl5 = Label(topFrame, text='Education', padx=15, pady=15, width=20,font=("bold", 12) )
lbl5.grid(column=1, row=2)
ent2= Entry(topFrame, width=40)
ent2.grid(column=2, row=2)

lbl6 = Label(topFrame, text='Experience', padx=15, pady=15, width=20,font=("bold", 12) )
lbl6.grid(column=1, row=3)
ent3= Entry(topFrame, width=40)
ent3.grid(column=2, row=3)

lbl2 = Label(topFrame, text='Key Skills', padx=15, pady=15, width=20,font=("bold", 12) )
lbl2.grid(column=1, row=4)
textbox= Text(topFrame, width=30, height=10, wrap=WORD, padx=5 )
textbox.grid(column=2, row=4)

lbl3 = Label(topFrame, text='Add CV', padx=15, pady=15, width=20, font=("bold", 12))
lbl3.grid(row=5, column=1)
btn1 = Button(topFrame, text="Choose a file",  command=open_file )
btn1.grid(column=2, row=5)
btn2 = Button(topFrame, text="Analse",  command=analyze )
btn2.grid(column=2, row=7)
lbl4=Label(topFrame, width=25)
lbl4.grid(column=2, row=6)

window.mainloop()
"""
"""
train_set = ["The sky is blue.", "The sun is bright."]  # Documents
test_set = ["The sun in the sky is bright."]  # Query
stopWords = stopwords.words('english')

vectorizer = CountVectorizer(stop_words = stopWords)
#print (vectorizer)
transformer = TfidfTransformer()
#print transformer

trainVectorizerArray = vectorizer.fit_transform(train_set).toarray()
testVectorizerArray = vectorizer.transform(test_set).toarray()
print ('Fit Vectorizer to train set', trainVectorizerArray)
print ('Transform Vectorizer to test set', testVectorizerArray)


cx = lambda a, b : round(np.inner(a, b)/(LA.norm(a)*LA.norm(b)), 3)

for vector in trainVectorizerArray:
    print (vector)
    for testV in testVectorizerArray:
        print (testV)
        cosine = cx(vector, testV)
        print (cosine)
transformer.fit(trainVectorizerArray)

print("\n")
print (transformer.transform(trainVectorizerArray).toarray())

transformer.fit(testVectorizerArray)
print("\n")
tfidf = transformer.transform(testVectorizerArray)
print ("Similarity result:", tfidf.todense())


raw_documents = ["I am taking the show on the road.",
                 "My socks are a force multiplier.",
             "I am the barber who cuts everyone's hair who doesn't cut their own.",
             "Legend has it that the mind is a mad monkey.",
            "I make my own fun."]
print("Number of documents:",len(raw_documents))

gen_docs = [[w.lower() for w in word_tokenize(text)]
            for text in raw_documents]
#print(gen_docs)

dictionary = gensim.corpora.Dictionary(gen_docs)
#print(dictionary[5])
#print(dictionary.token2id['road'])
#print("Number of words in dictionary:",len(dictionary))
#for i in range(len(dictionary)):
    #print(i, dictionary[i])

corpus = [dictionary.doc2bow(gen_doc) for gen_doc in gen_docs]
#print(corpus)
tf_idf = gensim.models.TfidfModel(corpus)
#print(tf_idf)
s = 0
for i in corpus:
    s += len(i)
#print(s)
sims = gensim.similarities.Similarity('C:\Similarity\sims',tf_idf[corpus],
                                      num_features=len(dictionary))
query_doc = [w.lower() for w in word_tokenize("Socks are a force for good.")]
#print(query_doc)
query_doc_bow = dictionary.doc2bow(query_doc)
#print(query_doc_bow)
query_doc_tf_idf = tf_idf[query_doc_bow]
#sims[query_doc_tf_idf]
print(query_doc_tf_idf)
print('Query result:', sims[query_doc_tf_idf])
"""

