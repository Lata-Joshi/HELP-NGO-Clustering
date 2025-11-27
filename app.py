import streamlit as st
import numpy as np
import pandas as pd
import joblib

# loading the joblib instances over here
with open('pipeline.joblib','rb') as file: # here rb is read binary
    preprocess = joblib.load(file)

# loading model
with open('model.joblib','rb') as file:
    model = joblib.load(file)

# Creating title and subtitle
st.title('HELP NGO ORGANIZATION')
st.subheader('This application will help to identify the development category of a country using socio-economic factors. Original data has been clustered using Kmeans.')

# taking input from user
gpp = st.number_input('Enter the GDPP of a country(GDP per population)')
income = st.number_input('Enter Income per population')
exports = st.number_input('Exports of goods and services per capita. Given as %age of the GDP per capita')
imports = st.number_input('Imports of goods and services per capita. Given as %age of the GDP per capita')
inflation = st.number_input('Inflation : The measurement of the annual growth rate of the Total GDP')
life_exp = st.number_input('Life Expectency : The average number of years a new born child would live if the current mortality patterns are to remain the same')
child_mort = st.number_input('Child Mortality : Death of children under 5 years of age per 1000 live births')
health = st.number_input('Health : Total health spending per capita. Given as %age of GDP per capita')
total_fert = st.number_input('Total Fertility : The number of children that would be born to each woman if the current age-fertility rates remain the same')

# create the input list
input_list = [child_mort,exports,health,imports,inflation,income,life_exp,total_fert,gpp]

# create final input data
final_input_list = preprocess.transform([input_list])

# now we have to do prediction
if st.button('predict'):
    prediction = model.predict(final_input_list)[0]
    if prediction ==0:
        st.success('Developing')
    elif prediction ==1:
        st.success('Developed')
    else:
        st.error('Under Developed')
