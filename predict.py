import pandas as pd
import numpy as np
import sys
from patsy import dmatrices
import tensorflow as tf

from flask import Flask, render_template

app = Flask(__name__)

from sklearn.externals import joblib
decs_tree = joblib.load('models/larh_decision_t.model')
rand_f = joblib.load('models/larh_random_forest.model')
log_model = joblib.load('models/larh_logistic_reg.model')

#load neural net
df = pd.read_csv('train_data_x1.csv',header=0)
df['win_pct_diff'] = df['A_PCT'] - df['B_PCT']
df['rpi_diff'] = df['A_RPI'] - df['B_RPI']
df['team_rating_diff'] = df['TEAM_RATING'] - df['TEAM_RATING_1']
from keras.utils import np_utils

y, X = dmatrices('TEAMRESULT ~ rpi_diff + team_rating_diff + A_RPI + B_RPI + win_pct_diff +\
                A_RPI_C + B_RPI_C + TEAM_SEED_1 + TEAM_SEED + RD7_WIN + RD7_WIN_1 - 1',
                df, return_type='dataframe')

X = X.values
y = y.values

dimof_input = X.shape[1]
dimof_output = len(set(y.flat))

y_test = np.ravel(y)
# Set y categorical
y = np_utils.to_categorical(y, dimof_output)
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.optimizers import SGD

model = Sequential()

# Set constants

dimof_middle = 100
dropout = 0.5

verbose = 1
"""MLP for classification (0,1)"""

model.add(Dense(dimof_middle, input_dim=dimof_input, init="uniform", activation='relu' ))
model.add(Dropout(dropout))
model.add(Dense(dimof_middle, init="uniform", activation='relu' ))
model.add(Dropout(dropout))
model.add(Dense(dimof_output, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop')
countof_epoch = 10
batch_size = 64

model.fit(
    X, y,
    show_accuracy=True,
    batch_size=batch_size, nb_epoch=countof_epoch, verbose=verbose)

loss, accuracy = model.evaluate(X, y, show_accuracy=True, verbose=verbose)
print('loss: ', loss)
print('accuracy: ', accuracy)



#load data
team_info = pd.read_csv('real_data.csv', header=0, index_col = 7)
team_info_verse = pd.read_csv('verse_data.csv', header=0, index_col = 0)

def create_test_dataset(team1, team2):
    data = []
    data.append(team_info.loc[team1, 'RPI'] - team_info.loc[team2, 'RPI']) #RPI diff
    data.append(team_info.loc[team1, 'TEAM_RATING'] - team_info.loc[team2, 'TEAM_RATING']) #TEAM RATING
    data.append(team_info.loc[team1, 'RPI']) #A_RPI
    data.append(team_info.loc[team2, 'RPI']) #B_RPI
    data.append(team_info.loc[team1, 'PCT'] - team_info.loc[team2, 'PCT']) #TEAM RATING
    data.append(team_info.loc[team1, 'RPI_C']) #A_RPI_C
    data.append(team_info.loc[team2, 'RPI_C']) #B_RPI_C
    data.append(team_info.loc[team2, 'TEAM_SEED']) #TEAM_SEED 2
    data.append(team_info.loc[team1, 'TEAM_SEED']) #TEAM_SEED
    data.append(team_info.loc[team1, 'RD7_WIN']) # WIN_F 2
    data.append(team_info.loc[team2, 'RD7_WIN']) # WIN_F 1
    return data


def print_who_win(model, result, team1, team2):
    print result
    team_won = team1 if result[0][0] < result[0][1] else team2
    prob_won = result[0][1] if result[0][0] < result[0][1] else result[0][0]

    p_str = model + ' predict : ' + team_won + '<br>'

    p_str = p_str + "<b> probability : {0} %".format(int(prob_won * 100.00))
    return p_str + '</b>'

def predict(team1,team2):
    try:
        t1 =  create_test_dataset(team1, team2) # first predict
        response = print_who_win('Logistic regression: ', log_model.predict_proba(t1), team1, team2) + '<br>'
        response = response + print_who_win('Neural Net: ', model.predict_proba(np.array([t1])), team1, team2) + '<br>'

        return response
    except:
        e = sys.exc_info()[0]
        return ("<p>Error: %s</p>" % e )

@app.route('/')
def index():
    return render_template('predict.html')

@app.route('/predict/<team1>/<team2>')
def make_predict(team1,team2):
    team1 = team1.replace('_',' ')
    team2 = team2.replace('_',' ')
    return predict(team1,team2)

if __name__ == '__main__':
    app.run(debug=True)
