# Diabetes Prediction Web Application

## Project Description
This project aims to predict the presence of diabetes in women using machine learning. It was developed as part of the Applied MSc in Data Analytics & AI at DSTI. The final model uses CatBoost and is deployed in a web application built with Flask.

The web application allows:
- **Manual prediction**: Users fill in a form with 8 medical inputs (Pregnancies, Glucose, BMI, etc.).
- **Batch prediction**: Users drag-and-drop a CSV file to get predictions for multiple entries.

---

##Project Structure
- `WEBSITE/` → All web application files, including the trained model and web interface code.de
- `Diabetes_Notebook_EDA_and_ML - WITH AUC -ROC.ipynb` → Data exploration and model building (Jupyter notebook).
- `Project - Predicting diabetes outcome for women.pdf`→ The project group report.
- `README.md` → This documentation.
- `REQUIREMENTS.txt` → Python libraries needed to run the app.
- `TAIPEI_diabetes.csv` → Dataset used for training and testing.


---

##Model
- The final model is based on the **CatBoost** algorithm.
- It was trained and evaluated using metrics such as Accuracy, Recall, and AUC.

---

##Installation & Execution

1- Clone the repository

git clone https://github.com/Vanes-sa03/diabetes-prediction-ml.git
cd diabetes-prediction-ml

2- Install required libraries
Make sure you have Python 3 and Anaconda installed.

From an Anaconda Prompt:
`pip install -r REQUIREMENTS.txt`

3- Run the web application
Go into the WEBSITE folder and launch the app: (you will find a README and REQUIREMENTS for the website)
`cd WEBSITE
python website.py`

4- Open the app in your browser
Go to: http://127.0.0.1:5000

