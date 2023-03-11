import pandas as pd

stock_data = pd.read_csv('Data/NFLX.csv', index_col='Date')
# data['Date'] = pd.to_datetime(data.Date, format='%Y/%m/%d')

# print("Số lượng dòng dữ liệu: ", data.shape[0])
# print("Số lượng cột dữ liệu: ", data.shape[1])
print(stock_data.head())
# print(data.info())
# print(data.describe())

# import matplotlib.pyplot as plt
# plt.figure(figsize=(15, 7))
# plt.title("Biểu đồ trực quan hóa dữ liệu giá mở cửa")
# plt.plot(data["Date"], data['Open'], color="red")
# plt.xlabel("Thời gian")
# plt.ylabel("Giá")
# plt.grid(color = 'green', linestyle = '--', linewidth = 0.5)
# plt.show()


# plt.figure(figsize=(15, 7))
# plt.title("Biểu đồ trực quan hóa dữ liệu giá cao")
# plt.plot(data["Date"], data['High'], color="green")
# plt.xlabel("Thời gian")
# plt.ylabel("Giá")
# plt.grid(color = 'green', linestyle = '--', linewidth = 0.5)
# plt.show()

# plt.figure(figsize=(15, 7))
# plt.title("Biểu đồ trực quan hóa dữ liệu giá thấp")
# plt.plot(data["Date"], data['Low'], color="yellow")
# plt.xlabel("Thời gian")
# plt.ylabel("Giá")
# plt.grid(color = 'green', linestyle = '--', linewidth = 0.5)
# plt.show()

# plt.figure(figsize=(15, 7))
# plt.title("Biểu đồ trực quan hóa dữ liệu giá đóng cửa")
# plt.plot(data["Date"], data['Close'], color="purple")
# plt.xlabel("Thời gian")
# plt.ylabel("Giá")
# plt.grid(color = 'green', linestyle = '--', linewidth = 0.5)
# plt.show()

# plt.figure(figsize=(15, 7))
# plt.title("Biểu đồ trực quan hóa dữ liệu giá đóng cửa điều chỉnh")
# plt.plot(data["Date"], data['Adj Close'], color="blue")
# plt.xlabel("Thời gian")
# plt.ylabel("Giá")
# plt.grid(color = 'green', linestyle = '--', linewidth = 0.5)
# plt.show()

# Sử dụng dữ liệu giá của các cột Open, High, Low ==> Dự đoán giá trị đóng cửa cho ngày tiếp theo
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping # Tương tự như với ModelCheckpoint, khi mô hình học đến độ chính xác cao nhất có thể sẽ tự dừng chương trình, tránh trường hợp overfitting
from sklearn.preprocessing import MinMaxScaler, StandardScaler # StandardScaler giúp làm giảm tỉ lệ tiêu chuẩn của dữ liệu thông qua việc đổi các giá trị đặc trưng trong dữ liệu sao cho chúng có giá trị trung bình bằng 0 và độ lệch chuẩn bằng 1
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split, TimeSeriesSplit # Chia dữ liệu cho các bài toán theo chuỗi thời gian
import numpy as np

target_y = stock_data['Close']
x_feat = stock_data.iloc[:,0:1]
print(x_feat)
print('Kết quả\n')
print(x_feat.shape)

sc = StandardScaler()
X_ft = sc.fit_transform(x_feat.values)
X_ft = pd.DataFrame(columns=x_feat.columns, data=X_ft, index=x_feat.index)

def LSTM_SPLIT(data, n_steps):
  X, y = [], []
  for i in range(len(data) - n_steps + 1):
    X.append(data[i:i + n_steps, :-1])
    y.append(data[i + n_steps - 1, -1])
  return np.array(X), np.array(y)


X1, y1 = LSTM_SPLIT(x_feat.values, 2)
train_slipt = 0.8
split_idx = int(np.ceil(len(X1)*train_slipt))
data_index = x_feat.index

X_train, X_test = X1[:split_idx], X1[split_idx:]
y_train, y_test = y1[:split_idx], y1[split_idx:]
X_train_date, X_test_date = data_index[:split_idx], data_index[split_idx:]

print("Dữ liệu của X1.shape = ", X1.shape)
print("Dữ liệu của X_train.shape = ", X_train.shape)
print("Dữ liệu của X_test.shape = ", X_test.shape)
print("Dữ liệu của y_test.shape = ", y_test.shape)

lstm = Sequential()
lstm.add(LSTM(32, input_shape=(X_train.shape[1], X_train.shape[2]), activation='relu', return_sequences=True))
lstm.add(Dense(1))
lstm.compile(loss='mean_squared_error', optimizer='adam')
lstm.summary()
history = lstm.fit(X_train, y_train, epochs=5, batch_size=4, verbose=2, shuffle=False)
lstm.save('Data/OXY.hdf5')
import matplotlib.pyplot as plt
plt.figure(1)
plt.plot(history.history['loss'], label='Loss')
plt.title('Biểu đồ trực quan hóa sự mất mát trong quá trình huấn luyện')
plt.legend()
plt.show()

y_predict = lstm.predict(X_test)
rmse = mean_squared_error(y_test, y_predict, squared=False)
mape = mean_absolute_percentage_error(y_test, y_predict)
print("RMSE: ", rmse)
print("MAPE: ", mape)