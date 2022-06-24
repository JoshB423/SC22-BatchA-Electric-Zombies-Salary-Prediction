# import requirements needed
from flask import Flask , render_template , request , redirect , url_for
import pandas as pd
import numpy as np
import json
import plotly
import plotly.express as px
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import plotly.graph_objs as go
import pickle
from sklearn.model_selection import train_test_split
from utils import get_base_url

# setup the webserver
# port may need to be changed if there are multiple flask servers running on same server
port = 12349
base_url = get_base_url(port)

# if the base url is not empty, then the server is running in development, and we need to specify the static folder so that the static files are served
if base_url == '/':
    app = Flask(__name__)
else:
    app = Flask(__name__, static_url_path=base_url+'static')

    
df_salary = pd.read_csv("https://raw.githubusercontent.com/VishnuNelapati/Junk/main/salary.csv")
df_salary = df_salary.replace(to_replace="?" ,value=np.nan) 
from sklearn.impute import SimpleImputer
Imputer = SimpleImputer(strategy="most_frequent")
df_salary[['workclass']]=Imputer.fit_transform(df_salary[['workclass']])
df_salary[['occupation']]=Imputer.fit_transform(df_salary[['occupation']])
df_salary[['native_country']]=Imputer.fit_transform(df_salary[['native_country']])
df_salary = df_salary.drop(['fnlwgt', 'capital_gain', 'capital_loss'],axis = 1)
df_salary['native_country'] = df_salary['native_country'].apply(lambda x : "United-States" if x == "United-States" else "others")
df_salary = pd.get_dummies(df_salary , columns = ['sex','race',"occupation","workclass","relationship","native_country","salary","marital_status",],drop_first=True)
useless_information = ['education']
df_salary = df_salary.drop(useless_information, axis = 1)
X = df_salary.drop("salary_>50K",axis = 1)
columns = X.columns


def encoded_data(data , columns = columns):
    data_list = ['age' ,'workclass' , 'education_num' , 'marital_status' , 'occupation' ,'relationship' ,'race' ,'sex' ,'hours_per_week' ,'native_country']
    numeric_list = []
    categorical_list = []
    for idx,i in enumerate(data):
      try:
        int(i)
        numeric_list.append(int(i))
      except:
        categorical_list.append(f'{data_list[idx]}_{i}')

        
    for i in columns[3:]:
      if i in categorical_list:
        numeric_list.append(1)
      else:
        numeric_list.append(0)

    return numeric_list
    

# set up the routes and logic for the webserver
@app.route(f'{base_url}'  , methods = ["GET","POST"])
def home():
    if request.method == "POST":
        values = [i for i in request.form.values()]
        print(values)
        print(encoded_data(values))
        
        loaded_model = pickle.load(open("neural_network_model.sav", 'rb'))
        result = loaded_model.predict(np.array(encoded_data(values)).reshape(1,40))
        salary = "less than 50K" if result[0] == 0 else "greater than 50k"
        print(salary)
        
        html_df = pd.DataFrame(values).T
        html_df.columns = ['age' ,'workclass' , 'education_num' , 'marital_status' , 'occupation' ,'relationship' ,'race' ,'sex' ,'hours_per_week' ,'native_country']
        
        df_html = html_df.to_html(classes="table table-dark")
        pred = f"With the given attributes the individual salary earned might be {salary}."
    
        return render_template('index.html' , values = pred ,
                               df_html = df_html)
    return render_template('index.html')

# define additional routes here
# for example:
# @app.route(f'{base_url}/team_members')
# def team_members():
#     return render_template('team_members.html') # would need to actually make this page

if __name__ == '__main__':
    # IMPORTANT: change url to the site where you are editing this file.
    website_url = 'cocalc20.ai-camp.dev/'
    
    print(f'Try to open\n\n    https://{website_url}' + base_url + '\n\n')
    app.run(host = '0.0.0.0', port=port, debug=True)
