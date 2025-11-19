import flask
print(flask.__version__)

from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)
app.env = "development"



def column_name_replace(df):
    df = df.copy()
    df.rename(columns={
        'Saving accounts': 'Saving_accounts',
        'Checking account': 'Checking_account',
        'Credit amount': 'Credit_amount'
    }, inplace=True)
    return df



model = joblib.load("pipeline.joblib")


@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():

    if request.method == 'POST':

     
        Age = int(request.form['Age'])
        Sex = request.form['Sex']
        Job = int(request.form['Job'])
        Housing = request.form['Housing']
        Saving_accounts = request.form['Saving_accounts']
        Checking_account = request.form['Checking_account']
        Credit_amount = int(request.form['Credit_amount'])
        Duration = int(request.form['Duration'])
        Purpose = request.form['Purpose']

      
        data_dict = {
            "Age": [Age],
            "Sex": [Sex],
            "Job": [Job],
            "Housing": [Housing],
            "Saving accounts": [Saving_accounts],
            "Checking account": [Checking_account],
            "Credit amount": [Credit_amount],
            "Duration": [Duration],
            "Purpose": [Purpose]
        }

        test_df = pd.DataFrame(data_dict)

        print("Incoming Test DF:")
        print(test_df)

        prediction = model.predict(test_df)[0]

        predicted = "Risky" if prediction == 1 else "No Risk"

        return render_template(
            "index.html",
            result=f"The model has predicted: {predicted}"
        )


if __name__ == "__main__":
    app.run(host='localhost', port=5001, debug=True)

