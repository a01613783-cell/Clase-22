import numpy as np
import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier


st.write(''' # Predicción de condición física (¿Está en forma?) ''')
st.image("fitnessf.png", caption="Evaluación de condición física a partir de indicadores biométricos y hábitos diarios.")

st.header('Datos de evaluación')

def user_input_features():
  # Entrada
  age = st.number_input('Edad (años):', min_value=18, max_value=100, value=30, step=1)
  height_cm = st.number_input('Estatura (cm):', min_value=120, max_value=230, value=170, step=1)
  weight_kg = st.number_input('Peso (kg):', min_value=30, max_value=250, value=70, step=1)
  heart_rate = st.number_input('Frecuencia cardíaca (lpm):', min_value=40.0, max_value=200.0, value=70.0, step=0.1)
  blood_pressure = st.number_input('Presión arterial sistólica (mmHg):', min_value=80.0, max_value=220.0, value=120.0, step=0.1)
  sleep_hours = st.number_input('Horas de sueño diarias:', min_value=1.0, max_value=16.0, value=7.5, step=0.1)
  nutrition_quality = st.number_input('Calidad de nutrición (0 a 10):', min_value=0.0, max_value=10.0, value=5.0, step=0.1)
  activity_index = st.number_input('Índice de actividad física (1 a 5):', min_value=1.0, max_value=5.0, value=3.0, step=0.1)
  smokes = st.number_input('¿Fuma? (0 = No, 1 = Sí):', min_value=0, max_value=1, value=0, step=1)
  gender = st.number_input('Género (0 = Femenino, 1 = Masculino):', min_value=0, max_value=1, value=0, step=1)

  user_input_data = {'age': age,
                     'height_cm': height_cm,
                     'weight_kg': weight_kg,
                     'heart_rate': heart_rate,
                     'blood_pressure': blood_pressure,
                     'sleep_hours': sleep_hours,
                     'nutrition_quality': nutrition_quality,
                     'activity_index': activity_index,
                     'smokes': smokes,
                     'gender': gender}

  features = pd.DataFrame(user_input_data, index=[0])

  return features

df = user_input_features()

fitness = pd.read_csv('Fitness_Classification_Limpio.csv', encoding='utf-8')
X = fitness.drop(columns='is_fit')
Y = fitness['is_fit']

classifier = DecisionTreeClassifier(max_depth=4, criterion='entropy', min_samples_leaf=25, max_features=9, random_state=6)
classifier.fit(X, Y)

prediction = classifier.predict(df)

st.subheader('Predicción')
if prediction == 0:
  st.write('No está en forma')
elif prediction == 1:
  st.write('Sí está en forma')
else:
  st.write('Sin predicción')
