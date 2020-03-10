from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing
from multiprocessing import Pool
import numpy as np
import pandas_datareader.data as web
import pandas as pd

def modelLSMT(df,tick):
    df = df.sort_index(ascending=True, axis=0)
    df=df.reset_index()
    df = df.loc[:,[tick,'Date']]
    new_data = pd.DataFrame(index=range(0,len(df)),columns=['Date', 'Close'])
    new_data['Date'] = df['Date']
    new_data['Close'] = df[tick]
    new_data.index = new_data.Date
    new_data.drop('Date', axis=1, inplace=True)
    dataset = new_data.values

    train = dataset[0:(dataset.shape[0]-252),:]
    valid = dataset[(dataset.shape[0]-252):,:]

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)

    x_train, y_train = [], []
    for i in range(60,len(train)):
        x_train.append(scaled_data[i-60:i-1,0])
        y_train.append(scaled_data[i,0])
    x_train, y_train = np.array(x_train), np.array(y_train)

    x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))

    # create the LSTM Model
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1],1)))
    model.add(LSTM(units=50))
    model.add(Dense(1))

    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(x_train, y_train, epochs=1, batch_size=1, verbose=1)

    
    inputs = new_data[len(new_data) - len(valid) - 60:].values
    inputs = inputs.reshape(-1,1)
    inputs  = scaler.transform(inputs)

    
    X_test = []
    for i in range(60,inputs.shape[0]):
        X_test.append(inputs[i-60:i-1,0])
    X_test = np.array(X_test)

    X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))
    closing_price = model.predict(X_test)
    closing_price = scaler.inverse_transform(closing_price)

    rms=np.sqrt(np.mean(np.power((valid-closing_price),2)))
    print("RMS ERROR: ",rms)
    train = new_data[0:(dataset.shape[0]-252)]
    valid = new_data[(dataset.shape[0]-252):]
    valid['Predictions'] = closing_price[:,0]
    #plt.plot(train['Close'])
    #plt.plot(valid[['Close','Predictions']])
    print("done")
    return(model,valid,train,new_data)
