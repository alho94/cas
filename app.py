import numpy as np
import pandas as pd
import streamlit as st
from joblib import load

import csv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score


from joblib import load, dump

try:
  clf = load('model.joblib') 
except:


  df = pd.read_csv("data.csv", sep=',' ,escapechar="\\",quoting=csv.QUOTE_NONE)

  df = df[df["Unique class"]!="keine Klasse"]

  df["class_num"] = pd.Series(pd.factorize(df["Unique class"])[0], dtype='int32').values
  df = df.dropna()
  print(df["class_num"].value_counts())


  train_text, test_text, train_labels, test_labels = train_test_split(df['Sentence'], df['class_num'], 
                                                                      random_state=1234, 
                                                                      test_size=0.2, 
                                                                      stratify=df['class_num'])

  vectorizer = TfidfVectorizer()
  X_train = vectorizer.fit_transform(train_text)
  X_test = vectorizer.transform(test_text)

  clf = RandomForestClassifier(bootstrap=False,
   min_samples_leaf=2,
   min_samples_split=8,
   n_estimators= 1255)
  clf.fit(X_train, train_labels)
  dump(clf, 'model.joblib') 
  dump(vectorizer, 'vectorizer.joblib')











clf = load('model.joblib') 
vectorizer = load('vectorizer.joblib') 


st.title('Klassifizierung medizinischer Literatur')
st.write("Dieses Program dient als Proof-of-Concept für das CAS Practical Machine Learning.")


def mypredict(mylist):

    classes = ["Drug Discovery", "Preclinical","Clinical Research","Pharmacovigilance"]
    vectorized_list = vectorizer.transform(mylist)
    
    pred = []
    for satz in vectorized_list:
      p = clf.predict(satz)[0]
      print(p)
      pred.append(classes[p])
    pred_df = pd.DataFrame({
        "Satz" : mylist,
        "Prediction" : pred
      })
      
    return pred_df






txt = st.text_area('Text eingeben:', '''Ten adverse events were reported.
No serious adverse event were reported.
New molecolues were discovered.
Various serious adverse drug reactions were reporeted by patients.
We have received some case reports of hospitalized patienst.
animal.
Two female baboons died after administration of X.
Case IV studies have proven X.''', height = 300)

txt = txt.split(".")
txt = [t for t in txt if len(t)>3]
pred = mypredict(txt)
print(pred)

st.write("**Klasse für gesamten Text:**")
st.bar_chart( pred["Prediction"].value_counts())



clicked_id=0
 # # Show user table 
colms = st.columns((1, 4, 2, 1))
fields = ["ID", 'Satz', 'Vorhersage',]
for col, field_name in zip(colms, fields):
    # header
    col.write("**"+field_name+"**")
for x, email in enumerate(pred['Satz']):
    col1, col2, col3, col4 = st.columns((1, 4, 2, 1))
    col1.write(x)  # index
    col2.write(pred['Satz'][x])  # email
    col3.write(pred['Prediction'][x])  # unique ID
    disable_status = True  # flexible type of button
    button_type = "Anpassen" if disable_status else "Block"
    button_phold = col4.empty()  # create a placeholder
    do_action = button_phold.button(button_type, key=x)
    if do_action:
         clicked_id = x
         pass # do some action with a row's data


classes = ["Drug Discovery", "Preclinical","Clinical Research","Pharmacovigilance"]

with st.expander(f"Feedback zu Nr. {clicked_id} geben"):
    st.write(f"""
    Satz:
    {pred['Satz'][clicked_id]}
    """)
    st.radio(
    "Prediction",
    ("Drug Discovery", "Preclinical","Clinical Research","Pharmacovigilance"),
    index=classes.index(pred["Prediction"][clicked_id]))



