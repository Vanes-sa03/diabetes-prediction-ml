"""
XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
Course Name :       Machine Learning with Python Labs' website
Project's Name :    Predicting diabetes outcome for women
Professor :         Hanna ABI AKL, DSTI
Instructor :        Christophe BECAVIN, PhD, Université Côte d’Azur
    --------------------------------------
Description:        Creates a web server using Flask and Flak_Dropzone
                    to make diabetes predictions with our fitted model.
    --------------------------------------
Student's group :   11
Authors:            Ronald LE PAPE
                    Vanessa GIRALDO VILLANUEVA
                    Almendra PEREZ
                    Shanchun YANG
                    Maruboina Sujeendra Kumar Reddy
    --------------------------------------        
Date:               2025-01-26
Version:            1.0
XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
"""



"""
--------------------------------------------------------------------------------------
->      Imports, configurations, initializations, global variables 
--------------------------------------------------------------------------------------
"""
# Imports
import os, time
import csv
import pandas as pd
import pickle
import catboost
from flask import Flask, render_template, request, redirect, url_for, send_file
from flask_dropzone import Dropzone
from colorama import Fore, Style

# Flask initialization
app = Flask(__name__)

# Flask-Dropzone config
basedir = os.path.abspath(os.path.dirname(__file__))

# Flask-Dropzone configuration update
app.config.update(
    UPLOADED_PATH=os.path.join(basedir, 'uploads'),
    # Flask-Dropzone config:
    DROPZONE_DEFAULT_MESSAGE="Drop your csv file here, then click Predict.",
    DROPZONE_ALLOWED_FILE_CUSTOM= True,
    DROPZONE_ALLOWED_FILE_TYPE='.csv',
    DROPZONE_MAX_FILE_SIZE=3,
    DROPZONE_MAX_FILES=1,
)

# Dropzone initialization
dropzone = Dropzone(app)


#global back_from_result
back_from_result    = 0

#global uploaded_file_path, uploaded_file_name and csv_results_file
uploaded_file_path  = ""
uploaded_file_name  = ""
csv_results_file    = ""



"""
--------------------------------------------------------------------------------------
->      Loading fitted ML model with Pickle 
--------------------------------------------------------------------------------------
"""
with open('catboost_model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)


"""
--------------------------------------------------------------------------------------
->      Defining path decorators (Flask) for the different forms / pages
        and associated code
--------------------------------------------------------------------------------------
"""
# Redirecting root page to input page
@app.route('/')
def home():
    return redirect('/input')

# Data input page
@app.route("/input", methods=["GET", "POST"])
def input():
    # POST means that a file is dropped in the drop zone
    if request.method == 'POST':
        
        global uploaded_file_path
        uploaded_file_path = ""

        global uploaded_file_name


        f = request.files.get('file')
        uploaded_file_path = os.path.join(app.config['UPLOADED_PATH'], f.filename)
        uploaded_file_name = f.filename
        #print(uploaded_file_path)
        f.save(uploaded_file_path)

        global back_from_result
        back_from_result = 0

    # GET means this page is called from a resutlts page's "Back to form" button
    elif request.method == 'GET':
        #global back_from_result
        back_from_result = 1
       
    return render_template("input_form.html")


# Unitarian/manual result page
@app.route("/result", methods=["POST"])
def result():
        
        # Getting the datas from the form
        donnees = request.form
        pregnancies                 = donnees.get('pregnancies')
        plasmaglucose               = donnees.get('plasmaglucose')
        diastolicbloodpressure      = donnees.get('diastolicbloodpressure')
        tricepsthickness            = donnees.get('tricepsthickness')
        seruminsulin                = donnees.get('seruminsulin')
        bmi                         = donnees.get('BMI')
        diabetespedigree            = donnees.get('diabetespedigree')
        age                         = donnees.get('age')        
    
        # Logging in  the server's window
        print(" ")
        print(Fore.YELLOW + "------------------------------------------------------------------------------------------------------------")
        print("UNITARIAN MANUAL MODE :")
        print(" ")
        print("loaded_model.__class__.__name__ : ", loaded_model.__class__.__name__)  # Ex: 'RandomForestClassifier'
        print(" ")
        # Creating a dataframe from the unitarian/manual form
        columns         =   ['pregnancies','plasmaglucose','diastolicbloodpressure','tricepsthickness','seruminsulin','bmi','diabetespedigree', 'age']
        unitarian_row   =   [int(pregnancies), int(plasmaglucose), int(diastolicbloodpressure), int(tricepsthickness), int(seruminsulin), float(bmi), float(diabetespedigree), int(age)]
        df_unitarian    =   pd.DataFrame([unitarian_row], columns=columns)
        
        # Logging in  the server's window
        print("Form values      -> ", unitarian_row)

        # Predicting, using the fitted model
        df_unitarian_prediction_proba = loaded_model.predict_proba(df_unitarian.values)
        
        # Logging in  the server's window
        print("Prediction proba -> [class0, class1]" + Fore.CYAN, df_unitarian_prediction_proba)
        print(Fore.YELLOW +"------------------------------------------------------------------------------------------------------------")
        print(Style.RESET_ALL)
        print(" ")

        # Rendering prediction
        for i, prob in enumerate(df_unitarian_prediction_proba):
            dominant_class  =   prob.argmax()
            confidence      =   (prob[dominant_class] * 100).round(decimals = 1)
        return render_template("result_form.html", prediction=dominant_class, probability = confidence)


# Batch result page
@app.route("/batchresult", methods=["GET", "POST"])
def batchresult():

    global uploaded_file_name
    global back_from_result


    if back_from_result == 1 :
       uploaded_file_name = ""

    if request.method == 'POST' and uploaded_file_name == "":
        return render_template("input_form.html")
    else:
        # Creating a data frame from the uploaded file :
        df_batch = pd.read_csv(uploaded_file_path,sep=",",index_col="PatientID")

        # Logging in  the server's window
        print(" ")
        print(Fore.YELLOW + "------------------------------------------------------------------------------------------------------------")
        print("BATCH MODE : ")
        print(" ")
        print("loaded_model.__class__.__name__ : ", loaded_model.__class__.__name__)  # Ex: 'RandomForestClassifier'
        print(" ")
        print("Uploaded file name              -> ",uploaded_file_name)
        print("Uploaded file columns count     -> ",df_batch.shape[1]+1) #adding 1 : to count the index column (PatientID)
        print("Uploaded file data record count -> ",df_batch.shape[0])
        print(" ")
        print(" ")

        # Predicting, using the fitted model
        array_batch_prediction = loaded_model.predict(df_batch.values)

        # Giving a name to the results columnby using rename function
        df_batch_prediction = pd.DataFrame(array_batch_prediction).rename(columns={0:"Diabetic"})

        # Concatenating df_batch and df_batch_prediction. 
        # The "set-axis" part forces the result'df to have the same index as df_batch, so the concat is successful (pandas uses it as a key to merge the 2 DFs)
        df_batch_results = pd.concat([df_batch, df_batch_prediction.set_axis(df_batch.index)], axis = 1)

        # Saving the results in a csv file :
        global csv_results_file
        csv_results_file = "batchresults.csv"
        df_batch_results.to_csv(csv_results_file, sep=',')

        # Read the CSV file and generate the HTML content
        with open(csv_results_file, mode='r', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            rows = list(reader)

            # Start building the HTML content
            html_content = """<table>"""

            # Add table rows
            for i, row in enumerate(rows):
                if i == 0:
                    # Add table headers
                    html_content += "<tr>" + "".join(f"<th>{cell}</th>" for cell in row) + "</tr>\n"
                elif i <= 10 :
                    # Add table data
                    html_content += "<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>\n"
                else :
                    number_of_rows = i
            # Close the table and HTML tags
            html_content += """</table>"""

        # Logging in the server :
        print("          -> processed", number_of_rows, "rows. Five first rows are shown below :")
        print(Fore.CYAN)
        print(df_batch_results.head(5))
        print(Fore.YELLOW + "------------------------------------------------------------------------------------------------------------")
        print(Style.RESET_ALL)
        print(" ")

    return render_template("batch_result_form.html", htmlTable = html_content)

# Page/URL associated with batch results file download link   
@app.route('/results_csv_download', methods=["GET", "POST"])
def results_csv_download():
    global csv_results_file

    file = csv_results_file
    return send_file(file, as_attachment=True) 




"""
--------------------------------------------------------------------------------------
# ->    RUNNING THE APP 
--------------------------------------------------------------------------------------
"""
# Starting-running the webapp
if __name__ == '__main__':
    app.run(debug=True)