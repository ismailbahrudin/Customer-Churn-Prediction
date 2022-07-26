
import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

st.write("""
 Simple Churn Customer Prediction App.
This app predicts the **Customer Churn** by their credit score, age, and tenure.
""")

st.sidebar.header('User Input Parameters')

def user_input_features():
    score = st.sidebar.slider('Credit score of the members', 350, 850, 500)
    age = st.sidebar.slider('Age of the member', 18, 100, 30)
    tenure = st.sidebar.slider('Tenure of the members', 0, 15, 7)
    data = {'score': score,
            'age': age,
            'tenure': tenure,}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

st.subheader('User Input parameters')
st.write(df)

churn = pd.read_csv('https://raw.githubusercontent.com/ismailbahrudin/Discount-Profit/3745beb6bba0461ce7644161839bfdbd99232625/data.csv')
X = churn.drop(['Row','Id','Surname','Nationality','Gender','Balance','Products','Card','Active','Salary','Exited'],axis=1)
Y = churn['Exited']

clf = RandomForestClassifier()
clf.fit(X, Y)

prediction = clf.predict(df)
prediction_proba = clf.predict_proba(df)

st.subheader('Class labels and their corresponding index number')
dataf= pd.DataFrame(['Not Churn','Churn',])
st.write(dataf)

st.subheader('Prediction')
#st.write(churn.target_names[prediction])
st.write(prediction)

if prediction==0:
 st.write('''Customer will not churn''')
elif prediction==1:
 st.write('''Customer will churn''')
st.subheader('Prediction Probability')
st.write(prediction_proba)
