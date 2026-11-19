import numpy as np
import streamlit as st
import pandas as pd

st.write(''' # Predicción de calificación de examen ''')
st.image("calificaciones.jfif", caption="Predicción de la calificación de un alumno.")

st.header('Datos del alumno')

def user_input_features():

  HorasEstudio = st.number_input('Horas de estudio:', min_value=0.0, max_value=24.0, value=0.0)
  HorasSueno = st.number_input('Horas de sueño:', min_value=0.0, max_value=24.0, value=0.0)
  Asistencia = st.number_input('Porcentaje de asistencia:', min_value=0.0, max_value=100.0, value=0.0)
  CalificacionPrevia = st.number_input('Calificación previa:', min_value=0, max_value=100, value=0)

  user_input_data = {'hours_studied': HorasEstudio,
                     'sleep_hours': HorasSueno,
                     'attendance_percent': Asistencia,
                     'previous_scores': CalificacionPrevia
                     }

  features = pd.DataFrame(user_input_data, index=[0])

  return features

df = user_input_features()
prediccion = 0

datos = pd.read_csv('student_exam_scores(3).csv')

X = datos.drop(columns=['student_id','exam_score'])
y = datos['exam_score']

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=2)

LR = LinearRegression()
LR.fit(X_train,y_train)

b1 = LR.coef_
b0 = LR.intercept_

prediccion = b0 + b1[0]*df['hours_studied'] + b1[1]*df['sleep_hours'] + b1[2]*df['attendance_percent'] + b1[3]*df['previous_scores']

st.subheader('Cálculo de la calificación')
st.write('La calificación estimada del alumno es ', prediccion)
