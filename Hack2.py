import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense

def split_sequences(sequences, n_steps):
    X, y = list(), list()
    for i in range(len(sequences)):
        end_ix = i + n_steps
        if end_ix > len(sequences):
            break
        seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix-1, -1]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

def result():
    result1 = pd.read_csv("result1.csv")
    result2 = pd.read_csv("result2.csv")
    result3 = pd.read_csv("result3.csv")
    result4 = pd.read_csv("result4.csv")
    result5 = pd.read_csv("result5.csv")
    result6 = pd.read_csv("result6.csv")
    result = pd.concat([result1, result2, result3, result4, result5, result6]).reset_index(drop=True)
    result.to_csv("result-3rd.csv", index= False)

def find_result(train, test):
    unique_course_id = train['Course_ID'].unique()
    result = pd.DataFrame()
    for i in range(len(unique_course_id)):
        if i < 0 or i >= 10:
            continue
        print("*************Course_ID************: ", unique_course_id[i])
        df_train = train[train['Course_ID'] == unique_course_id[i]].reset_index(drop=True)
        df_test = test[test['Course_ID'] == unique_course_id[i]].reset_index(drop=True)
        len_train = df_train.shape[0]
        len_test = df_test.shape[0]

        df_train['Log_User_Traffic'] = np.log(df_train['User_Traffic'])
        df_train = df_train[['ID', 'Day_No', 'Course_ID', 'Short_Promotion', 'Public_Holiday', 'Long_Promotion',
                             'Competition_Metric', 'Log_User_Traffic', 'Sales']]
        df_test = df_test[['ID', 'Day_No', 'Course_ID', 'Short_Promotion', 'Public_Holiday', 'Long_Promotion',
                           'Competition_Metric', 'User_Traffic']]
        df = pd.concat([df_train, df_test]).reset_index(drop=True)

        df = df[['Short_Promotion', 'Public_Holiday', 'Long_Promotion', 'Competition_Metric', 'Log_User_Traffic', 'Sales']]
        if len(df['Long_Promotion'].unique()) > 1 and len(df['Competition_Metric'].unique()) > 1:
            df = df[['Short_Promotion', 'Public_Holiday', 'Long_Promotion', 'Competition_Metric', 'Log_User_Traffic', 'Sales']]
        elif len(df['Long_Promotion'].unique()) > 1 and len(df['Competition_Metric'].unique()) == 1:
            df = df[['Short_Promotion', 'Public_Holiday', 'Long_Promotion', 'Log_User_Traffic', 'Sales']]
        elif len(df['Long_Promotion'].unique()) == 1 and len(df['Competition_Metric'].unique()) > 1:
            df = df[['Short_Promotion', 'Public_Holiday', 'Competition_Metric', 'Log_User_Traffic', 'Sales']]
        else:
            df = df[['Short_Promotion', 'Public_Holiday', 'Log_User_Traffic', 'Sales']]

        df = np.array(df)
        n_steps = 7
        X, y = split_sequences(df, n_steps)
        X_train = X[:-len_test]
        y_train = y[:-len_test]
        X_test = X[len_train - n_steps + 1:]
        n_features = X_train.shape[2]

        model = Sequential()
        model.add(LSTM(32, dropout=0.25, activation='relu', input_shape=(n_steps, n_features)))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        history = model.fit(X_train, y_train, validation_split=0.1 , nb_epoch=30, batch_size=64, verbose=2)

        # plt.plot(history.history['loss'])
        # plt.plot(history.history['val_loss'])
        # plt.title('Model loss')
        # plt.ylabel('Loss')
        # plt.xlabel('Epoch')
        # plt.legend(['Train', 'Test'], loc='upper left')
        # plt.show()

        y_pred = model.predict(X_test).astype(int).flatten()
        result_dict = {'ID': df_test['ID'], 'Sales': y_pred}
        result_dict = pd.DataFrame.from_dict(result_dict)
        result = pd.concat([result, result_dict])

    result.to_csv('result1.csv', index=False)

if __name__ == '__main__':
    train = pd.read_csv("train.csv")
    test  = pd.read_csv("test.csv")

    find_result(train, test)
    #result()
