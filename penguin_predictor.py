# Open Sublime text editor, create a new Python file, copy the following code in it and save it as 'penguin_app.py'.

# Importing the necessary libraries.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression  
from sklearn.ensemble import RandomForestClassifier

# Load the DataFrame
csv_file = 'penguin.csv'
df = pd.read_csv(csv_file)

# Display the first five rows of the DataFrame
df.head()

# Drop the NAN values
df = df.dropna()

# Add numeric column 'label' to resemble non numeric column 'species'
df['label'] = df['species'].map({'Adelie': 0, 'Chinstrap': 1, 'Gentoo':2})


# Convert the non-numeric column 'sex' to numeric in the DataFrame
df['sex'] = df['sex'].map({'Male':0,'Female':1})

# Convert the non-numeric column 'island' to numeric in the DataFrame
df['island'] = df['island'].map({'Biscoe': 0, 'Dream': 1, 'Torgersen':2})


# Create X and y variables
X = df[['island', 'bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g', 'sex']]
y = df['label']

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)


# Build a SVC model using the 'sklearn' module.
svc_model = SVC(kernel = 'linear')
svc_model.fit(x_train, y_train)
svc_score = svc_model.score(x_train, y_train)

# Build a LogisticRegression model using the 'sklearn' module.
log_reg = LogisticRegression()
log_reg.fit(x_train, y_train)
log_reg_score = log_reg.score(x_train, y_train)

# Build a RandomForestClassifier model using the 'sklearn' module.
rf_clf = RandomForestClassifier(n_jobs = -1)
rf_clf.fit(x_train, y_train)
rf_clf_score = rf_clf.score(x_train, y_train)

# Create a function that accepts 'model', island', 'bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g' and 'sex' as inputs and returns the species name.
@st.cache()
def prediction(model, island, bill_length_mm, bill_depth_mm, flipper_length_mm, body_mass_g, sex):
  if model.predict([[island, bill_length_mm, bill_depth_mm, flipper_length_mm, body_mass_g, sex]]) == 0:
    return "Adelie"
  elif model.predict([[island, bill_length_mm, bill_depth_mm, flipper_length_mm, body_mass_g, sex]]) == 1:
    return "Chinstrap"
  else:
    return "Gentoo"

# Design the App
# S2.3: Copy the following code and paste it in the 'improved_iris_app.py' file after the previous code.
# Add title widget
st.sidebar.title("Penguin Species Prediction App")
# Add 4 sliders and store the value returned by them in 4 separate variables. 

bill_length_input = st.sidebar.slider("Bill Length", float(df["bill_length_mm"].min()), float(df["bill_length_mm"].max()))
bill_depth_input = st.sidebar.slider("Bill Depth", float(df["bill_depth_mm"].min()), float(df["bill_depth_mm"].max()))
flipper_length_input = st.sidebar.slider("Bill Length", float(df["flipper_length_mm"].min()), float(df["flipper_length_mm"].max()))
body_mass_input = st.sidebar.slider("Body Mass", float(df["body_mass_g"].min()), float(df["body_mass_g"].max()))
island_input = st.sidebar.selectbox("Island: ", ("Biscoe", "Dream", "Torgersen"))

if island_input == "Biscoe":
  island_input = 0
elif island_input == "Dream":
  island_input = 1
elif island_input == "Torgersen":
  island_input = 2

sex_input = st.sidebar.selectbox("Sex: ", ("Male", "Female"))
if sex_input == "Male":
  sex_input = 0
elif sex_input == "Female":
  sex_input = 1

classifier = st.sidebar.selectbox("Classifier: ", ("Support Vector Machine", "Logistic Regression", "Random Forest Classifier"))
# When the 'Predict' button is clicked, check which classifier is chosen and call the 'prediction()' function.
# Store the predicted value in the 'species_type' variable accuracy score of the model in the 'score' variable. 
# Print the values of 'species_type' and 'score' variables using the 'st.text()' function.
if st.sidebar.button("Predict"):
  if classifier == "Support Vector Machine":
    penguin_species = prediction(svc_model, bill_length_input, bill_depth_input, flipper_length_input, body_mass_input, island_input, sex_input)
    penguin_score = svc_model.score(x_train, y_train)
  elif classifier == "Logistic Regression":
    penguin_species = prediction(log_reg, bill_length_input, bill_depth_input, flipper_length_input, body_mass_input, island_input, sex_input)
    penguin_score = log_reg.score(x_train, y_train)
  else:
    penguin_species = prediction(rf_clf, bill_length_input, bill_depth_input, flipper_length_input, body_mass_input, island_input, sex_input)
    penguin_score = rf_clf.score(x_train, y_train)
  st.write(f"Species Predicted: {penguin_species}\nScore: {penguin_score}")