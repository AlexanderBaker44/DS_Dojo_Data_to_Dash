import streamlit as st
import pickle as pkl
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

#open model file
model_file=open('model_folder/random_forest_model.pkl','rb')
#load model in
rf_model_import=pkl.load(model_file)
#close file
model_file.close()
#Let's recreate our data preprocessing pipeline
#differentiate labels to label, one hot, numerical, and output columns, this will help us use the data more easily
categorical_label_input=['Age Group','Episode_Month','Reported_Month']
categorical_one_hot_input=['Outbreak Associated','Neighbourhood Name','FSA','Source of Infection','Client Gender','Ever Hospitalized','Ever in ICU','Ever Intubated']
numerical_input=['episode_vs_reported_int']
output=['Outcome']
#read in the dataset
df = pd.read_csv('Data/toronto_covid_cleaned.csv')
#get the column names
column_list=list(df.columns)
#instantiate the one hot encoding model
ohe=OneHotEncoder()
#fit the one hot encoder model
ohe_model=ohe.fit(df[categorical_one_hot_input])
#Create the empty encoder and model dict
encoded_data=dict()
model_dict=dict()
#iterate through the cetegorical leveled fields
for i in categorical_label_input:
    #instantiate
    le=LabelEncoder()
    #fit the model on the data
    label_model=le.fit(df[i])
    #put models into a dictionary
    model_dict[i]=label_model

#Now that we have the data pipeline set up we
fields_columns_1=['Outbreak Associated','Source of Infection','Client Gender']
fields_columns_2=['Neighbourhood Name','FSA','Age Group']
fields_columns_3=['Ever Hospitalized','Ever in ICU','Ever Intubated']
fields_columns_4=['Episode_Month','Reported_Month','episode_vs_reported_int']
#spread out the columns
col1,col2,col3,col4 = st.columns(4)
#make a header for our dashboard
st.header('Lets Make Custom Dashboard Inputs')
#with the column, now we use our widgets
with col1:
    #radio widget execution
    outbreak=st.radio(fields_columns_1[0],list(set(df[fields_columns_1[0]])))
    #source widget execution
    source=st.selectbox(fields_columns_1[1],list(set(df[fields_columns_1[1]])))
    #gender widget execution
    gender=st.select_slider(fields_columns_1[2],list(set(df[fields_columns_1[2]])))

#now we execute the widgets in column 2
with col2:
    #execute neighborhood widget
    neighborhood=st.selectbox(fields_columns_2[0],list(set(df[fields_columns_2[0]])))
    #execute fsa widget
    fsa=st.selectbox(fields_columns_2[1],list(set(df[fields_columns_2[1]])))
    #if we want to sort the values of a categorical feature
    #create a list
    age_list=list(set(df[fields_columns_2[2]]))
    #execute sort function
    age_list.sort()
    #execute age widget
    age_group=st.select_slider(fields_columns_2[2],age_list)
#now let's create the thrid column
#say we want to edit the dashboard alues, this dict will do so
check_dict={True:'Yes',False:'No'}
with col3:
    #execute hospitalized widget
    ever_hosp=check_dict[st.checkbox(fields_columns_3[0])]
    #execute icu wdiget
    ever_in_icu=check_dict[st.checkbox(fields_columns_3[1])]
    #execute inutbate widget
    ever_intu=check_dict[st.checkbox(fields_columns_3[2])]
#execute the 4th column
with col4:
    #execute episode month widget
    episode_month=st.slider(fields_columns_4[0],min(df[fields_columns_4[0]]),max(df[fields_columns_4[0]]))
    #execute reported month widget
    reported_month=st.slider(fields_columns_4[1],min(df[fields_columns_4[0]]),max(df[fields_columns_4[0]]))
    #execute difference widget
    difference=st.slider(fields_columns_4[2],min(df[fields_columns_4[0]]),max(df[fields_columns_4[0]]))
#let's make the input from our widgets
input_list=[outbreak,source,gender,neighborhood,fsa,age_group,ever_hosp,ever_in_icu,ever_intu,episode_month,reported_month,difference]
#let's use the columns to edit the input
column_of_input=fields_columns_1+fields_columns_2+fields_columns_3+fields_columns_4
#Now let's make the inout into a dictionary
input_col_dict=dict(zip(column_of_input,input_list))
print(input_col_dict)


#now let's change the label input
lm_array=[]
for i in categorical_label_input:

    lm=model_dict[i]
    #execute the label encoder transform
    lm_data=lm.transform([input_col_dict[i]])
    lm_array.append(lm_data[0])


#now let's use the one hot encoding model on the onput
one_hot_data=[input_col_dict[i] for i in categorical_one_hot_input]

print(one_hot_data)
#execute the one hot transform
ohe_data=ohe_model.transform([one_hot_data])
dense_array=ohe_data.toarray()

#get feature names for the one hot encoded model
features=ohe.get_feature_names_out(categorical_one_hot_input)

print(dense_array)
#create the full input
inputs=list(dense_array[0])+lm_array
print(inputs)
#turn these into an array
inputs=np.array([inputs])
print(inputs)
#use model to make the prediction
number=rf_model_import.predict(inputs)[0]
#format the dashbaord display
display_dict={1:'Fatal',0:'Non-Fatal'}
#display the value
st.write(display_dict[number])
