import pandas as pd 
import math
import numpy as np
from sklearn import tree
from sklearn.model_selection import train_test_split 
import streamlit as st
from sklearn import preprocessing
from sklearn.metrics import  confusion_matrix
label_encoder = preprocessing.LabelEncoder()


test_data_df = pd.read_csv('data/clean_data_train.csv')
test_data_df.drop('registro', axis = 1, inplace=True)
test_data_df.drop('id', axis = 1, inplace=True)
test_data = test_data_df.dropna()
test_data['ninos'] = test_data['ninos'].astype(int)

st.title('Acomodation prediction using decision tree')

st.subheader('Analysys of dataset')
st.subheader('Training dataset')
st.write(test_data)

#Model training


encoder_genero = label_encoder.fit_transform(test_data['genero'])
#encoder_codigo_destino = label_encoder.fit_transform(test_data['codigo_destino'])
#columns = ['duracion_estadia', 'genero', 'edad', 'ninos', 'codigo_destino']
columns = ['duracion_estadia', 'genero', 'edad', 'ninos']
train_predictors = test_data[columns]
dummy_encoded_train_predictors = pd.get_dummies(train_predictors)
y = test_data['tipo_acomodacion']
x_features_one = dummy_encoded_train_predictors.values
x_train, x_validation, y_train, y_validation = train_test_split(x_features_one, y, test_size=0.20, random_state = 0)
tree_one = tree.DecisionTreeClassifier()
tree_one = tree_one.fit(x_train, y_train)
y_pred = tree_one.predict(x_validation)

st.subheader('Confusion matrix')
st.write(confusion_matrix(y_validation, y_pred))
tree_one_accuracy = math.ceil(round(tree_one.score(x_validation, y_validation), 4) * 100) 

st.subheader('Accuracy')
col1, col2 = st.columns(2)
col1.metric(label='Accuracy percentage', value = tree_one_accuracy)
col2.metric(label='Using logistic regression', value = 55)


#Second dataframe


prediction_df = pd.read_csv('data/DataAcomodacion.csv')
#columns_predictions = ['duracion_estadia', 'genero', 'edad', 'ninos', 'codigo_destino']
columns_predictions = ['duracion_estadia', 'genero', 'edad', 'ninos']
codigos_destino = ['US', 'ES', 'UK', 'AR', 'PE', 'NL','COL', 'IT']
ninos = [1,0]
average_age = prediction_df['edad'].mean()

prediction_df['edad'].fillna(average_age, inplace = True)
prediction_df['ninos'].fillna(np.random.choice(ninos), inplace = True)
prediction_df['codigo_destino'].fillna(np.random.choice(codigos_destino), inplace = True)
predicted_df = prediction_df[columns_predictions]
encoder_gennder_predictions = label_encoder.fit_transform(predicted_df['genero'])
#encoder_codigo_destino_predictions = label_encoder.fit_transform(predicted_df['codigo_destino'])
dummy_encoded_predictions = pd.get_dummies(predicted_df)
y_prediction = tree_one.predict(dummy_encoded_predictions)
new_df = prediction_df
new_df['tipo_acomodacion'] = y_prediction

st.subheader('Second data frame with predictions')
st.write(new_df)


#function to predict a result 

def get_prediction(data_frame):
    get_encoder_gennder_predictions = label_encoder.fit_transform(predicted_df['genero'])
    get_encoder_codigo_destino_predictions = label_encoder.fit_transform(predicted_df['codigo_destino'])
    get_y_dummy_encoded_predictions = pd.get_dummies(data_frame)
    get_y_prediction = tree_one.predict(get_y_dummy_encoded_predictions)
    st.write(get_y_prediction)




st.subheader('Get your prediction')

destination_option = st.selectbox(
     'Choose your destination',
     ('US', 'ES', 'UK', 'AR', 'PE', 'NL','COL', 'IT'))

number_of_days = math.floor(st.number_input('How many days you will be there?'))


age = math.floor(st.number_input('Age'))

gender = st.radio(
     "Whats your gender",
     ('M', 'F'))

kids = st.radio(
     "Kids",
     ('Yes', 'No'))

if kids == 'Yes':
    kids_number = 1
else:
    kids_number = 0

data_to_predict = {
    'duracion_estadia' : [number_of_days],
    'genero':[gender],
    'edad': [age],
    'ninos':[kids_number],
    'codigo_destino':[destination_option]
}

df_to_predict = pd.DataFrame(data_to_predict)
st.write(df_to_predict)


st.button(label='Get prediction')