import pandas as pd # Đọc dữ liệu từ tập dữ liệu
import matplotlib.pyplot as plt # Vẽ các biểu đồ đường dữ liệu
import numpy as np # Xử lý dữ liệu
import math
from sklearn.preprocessing import MinMaxScaler # Chuẩn hóa dữ liệu
from sklearn.metrics import r2_score, mean_absolute_error, mean_absolute_percentage_error # Đo mức độ phù hợp, sai số tuyệt đối trung bình và phần trăm sau số tuyệt đối trung bình
from keras.models import load_model, Sequential # Tải mô hình, tạo một INSTANCE của mô hình Neural network
from keras.callbacks import ModelCheckpoint # Lưu lại mô hình huấn luyện tốt nhất
from keras.layers import LSTM, Dense, Dropout

data = pd.read_csv('Data/Stock_price.csv')
data['Date'] = pd.to_datetime(data.Date, format='%m/%d/%Y')

# Hiển thị kích thước dữ liệu
print("Số lượng dòng dữ liệu: ", data.shape[0])
print("Số lượng cột dữ liệu: ", data.shape[1])
# Hiển thị dữ liệu 5 dòng đầu
print(data.head())
# Hiển thị kiểu dữ liệu của data
print(data.info())
# Hiển thị thông tin dữ liệu
print(data.describe())

# Vẽ biểu đồ trực quan hóa dữ liệu
# plt.figure(figsize=(15, 7)) # Xây dựng kích thước khung
# plt.title("Biểu đồ giá chứng khoán giá đóng cửa")
# plt.plot(data["Date"], data['Prev Close'], label="Giá đóng cửa", color="red")
# plt.xlabel("Thời gian")
# plt.ylabel("Giá")
# plt.grid(color = 'green', linestyle = '--', linewidth = 0.5)
# plt.show()

# plt.figure(figsize=(15, 7)) # Xây dựng kích thước khung
# plt.title("Biểu đồ giá chứng khoán giá mở cửa")
# plt.plot(data["Date"], data['Open'], label="Giá mở cửa", color="blue")
# plt.xlabel("Thời gian")
# plt.ylabel("Giá")
# plt.grid(color = 'green', linestyle = '--', linewidth = 0.5)
# plt.show()

# plt.figure(figsize=(15, 7)) # Xây dựng kích thước khung
# plt.title("Biểu đồ giá chứng khoán cao nhất")
# plt.plot(data["Date"], data['High'], label="Giá cao nhất", color="yellow")
# plt.xlabel("Thời gian")
# plt.ylabel("Giá")
# plt.grid(color = 'green', linestyle = '--', linewidth = 0.5)
# plt.show()

# plt.figure(figsize=(15, 7)) # Xây dựng kích thước khung
# plt.title("Biểu đồ giá chứng khoán thấp nhất")
# plt.plot(data["Date"], data['Low'], label="Giá thấp nhất", color="gray")
# plt.xlabel("Thời gian")
# plt.ylabel("Giá")
# plt.grid(color = 'green', linestyle = '--', linewidth = 0.5)
# plt.show()

# Cấu trúc lại khung dữ liệu dùng cho dự đoán
new_data = pd.DataFrame(data, columns=['Date', 'Prev Close', 'Open', 'High', 'Low'])
new_data.index = new_data.Date
new_data.drop('Date', axis=1, inplace=True)

# Tạo ra các khung chứa các giá trị dự đoán
# Dự đoán giá 'Prev Close' (Giá trước đóng cửa)
new_data_prediction_Prev_Close = new_data.filter(['Prev Close'])
# Dự đoán giá 'Open' (Giá mở cửa)
new_data_prediction_Open = new_data.filter(['Open'])
# Dự đoán giá 'High' (Giá cao nhất)
new_data_prediction_High = new_data.filter(['High'])
# Dự đoán giá 'Low' (Giá mở cửa)
new_data_prediction_Low = new_data.filter(['Low'])

# Chia dữu liệu thành 3 tập: Dữ liệu train, dữ liệu values và dữ liệu test. Trong đó: 80% làm dữ liệu train, còn lại được chia ra tiếp tục cho dữ liệu values và dữ liệu test
print("Độ dài ban đầu: ", len(new_data))
len_training_new_data = math.ceil(len(new_data)* .8)
print("Độ dài sau khi sử lý lấy dữ liệu train: ", len_training_new_data)
print("Kết quả dữ liệu tập train: ", len_training_new_data/len(new_data) * 100, "%")
train = new_data[:len_training_new_data]
# Số lượng dữ liệu ban đầu là 5075, dùng 80% (nghĩa là sử dụng 4060) dữ liệu cho train
# Còn lại: 5075 - 4060 = 1015 (Số lượng dữ liệu này được tiếp tục chia làm dữ liệu value và dữ liệu test)
len_values_test = len(new_data) - len(train)
print("Độ dài dữ liệu còn lại là: ", len_values_test)
# Lấy 80% dữ liệu còn lại làm dữ liệu values
len_training_new_values = math.ceil(len_values_test* .8)
print("Độ dài sau khi sử lý lấy dữ liệu values: ", len_training_new_values)
print("Kết quả dữ liệu tập values: ", (len_training_new_values/len_values_test) * 100, "%")
temp = new_data[len_training_new_data:]
value = temp[:len_training_new_values]
test = temp[len_training_new_values:]
print('=====================================')
print("Đây là dành cho tệp train\n", train)
print("Đây là dành cho tệp value\n", value)
print("Đây là dành cho tệp test\n", test)
print('------------------------------Kết quả------------------------------')
print('Số lượng dành cho tập train: ', len_training_new_data)
print('Số lượng dành cho tập values: ', len(value))
print('Số lượng dành cho tập test: ', len(test))
print('===================================================================')

# Tăng tốc độ xử lý của mô hình bằng cách chuẩn hóa lại dữ liệu
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data_Prev_Close = scaler.fit_transform(train['Prev Close'].values.reshape(-1,1))
scaled_data_Open = scaler.fit_transform(train['Open'].values.reshape(-1,1))
scaled_data_High = scaler.fit_transform(train['High'].values.reshape(-1,1))
scaled_data_Low = scaler.fit_transform(train['Low'].values.reshape(-1,1))

print('Độ dài của "Prev Close" là  : ', len(scaled_data_Prev_Close))
print('Độ dài của "Open" là        : ', len(scaled_data_Open))
print('Độ dài của "High" là        : ', len(scaled_data_High))
print('Độ dài của "Low" là         : ', len(scaled_data_Low))

# Tạo X_train và y_train ==> Mục tiêu: Dựa vào khoảng thời gian trước đó 60 ngày để có thể dự đoán được giá của 60 ngày tiếp theo
prediction_days = 60
X_train_Prev_Close, y_train_Prev_Close = [], []
for x in range(prediction_days, len(scaled_data_Prev_Close)):
  X_train_Prev_Close.append(scaled_data_Prev_Close[x-prediction_days:x, 0])
  y_train_Prev_Close.append(scaled_data_Prev_Close[x, 0])
X_train_Prev_Close, y_train_Prev_Close = np.array(X_train_Prev_Close), np.array(y_train_Prev_Close) # Xếp dữ liệu thành một mảng
X_train_Prev_Close = np.reshape(X_train_Prev_Close, (X_train_Prev_Close.shape[0], X_train_Prev_Close.shape[1], 1))
y_train_Prev_Close = np.reshape(y_train_Prev_Close, (y_train_Prev_Close.shape[0], 1))
print('Đây là dành cho train Prev_Close: ', X_train_Prev_Close.shape)

# X_train_Open, y_train_Open = [], []
# for x in range(prediction_days, len(scaled_data_Open)):
#   X_train_Open.append(scaled_data_Open[x-prediction_days:x, 0])
#   y_train_Open.append(scaled_data_Open[x, 0])
# X_train_Open, y_train_Open = np.array(X_train_Open), np.array(y_train_Open) # Xếp dữ liệu thành một mảng
# X_train_Open = np.reshape(X_train_Open, (X_train_Open.shape[0], X_train_Open.shape[1], 1))
# # y_train_Open = np.reshape(y_train_Open, (y_train_Open.shape[0], 1))
# print('Đây là dành cho train Open: ', X_train_Open.shape)

# X_train_High, y_train_High = [], []
# for x in range(prediction_days, len(scaled_data_High)):
#   X_train_High.append(scaled_data_High[x-prediction_days:x, 0])
#   y_train_High.append(scaled_data_High[x, 0])
# X_train_High, y_train_High = np.array(X_train_High), np.array(y_train_High) # Xếp dữ liệu thành một mảng
# X_train_High = np.reshape(X_train_High, (X_train_High.shape[0], X_train_High.shape[1], 1))
# # y_train_High = np.reshape(y_train_High, (y_train_High.shape[0], 1))
# print('Đây là dành cho train High: ', X_train_High.shape)


# X_train_Low, y_train_Low = [], []
# for x in range(prediction_days, len(scaled_data_Low)):
#   X_train_Low.append(scaled_data_Low[x-prediction_days:x, 0])
#   y_train_Low.append(scaled_data_Low[x, 0])
# X_train_Low, y_train_Low = np.array(X_train_Low), np.array(y_train_Low) # Xếp dữ liệu thành một mảng
# X_train_Low = np.reshape(X_train_Low, (X_train_Low.shape[0], X_train_Low.shape[1], 1))
# # y_train_Low = np.reshape(y_train_Low, (y_train_Low.shape[0], 1))
# print('Đây là dành cho train Low: ', X_train_Low.shape)


# Xây dựng mô hình LSTM
model_Prev_Close = Sequential()
model_Prev_Close.add(LSTM(units=128, return_sequences=True, input_shape = (X_train_Prev_Close.shape[1], 1)))
model_Prev_Close.add(LSTM(units=64))
model_Prev_Close.add(Dropout(0.5))
model_Prev_Close.add(Dense(units = 1))
model_Prev_Close.compile(optimizer = 'adam', loss = 'mean_squared_error')

# # Thực hiện huấn luyện mô hình
save_model_Prev_Close = 'Save_train/AI_Prev_Close.hdf5'
best_model_Prev_Close = ModelCheckpoint(save_model_Prev_Close, monitor='loss', verbose=2, save_best_only=True, mode='auto') # Tìm mô hình huấn luyện tốt nhất
model_Prev_Close.fit(X_train_Prev_Close, y_train_Prev_Close, epochs=100, batch_size=50, verbose=2, callbacks=[best_model_Prev_Close])

# Xây dựng mô hình LSTM
# model_Open = Sequential()
# model_Open.add(LSTM(units=128, return_sequences=True, input_shape = (X_train_Open.shape[1], 1)))
# model_Open.add(LSTM(units=64))
# model_Open.add(Dropout(0.5))
# model_Open.add(Dense(units = 1))
# model_Open.compile(optimizer = 'adam', loss = 'mean_squared_error')

# # Thực hiện huấn luyện mô hình
# save_model_Open = 'Save_train/AI_Open.hdf5'
# best_model_Open = ModelCheckpoint(save_model_Open, monitor='loss', verbose=2, save_best_only=True, mode='auto') # Tìm mô hình huấn luyện tốt nhất
# model_Open.fit(X_train_Open, y_train_Open, epochs=100, batch_size=50, verbose=2, callbacks=[best_model_Open])

# Xây dựng mô hình LSTM
# model_High = Sequential()
# model_High.add(LSTM(units=128, return_sequences=True, input_shape = (X_train_High.shape[1], 1)))
# model_High.add(LSTM(units=64))
# model_High.add(Dropout(0.5))
# model_High.add(Dense(units = 1))
# model_High.compile(optimizer = 'adam', loss = 'mean_squared_error')

# # Thực hiện huấn luyện mô hình
# save_model_High = 'Save_train/AI_High.hdf5'
# best_model_High = ModelCheckpoint(save_model_High, monitor='loss', verbose=2, save_best_only=True, mode='auto') # Tìm mô hình huấn luyện tốt nhất
# model_High.fit(X_train_High, y_train_High, epochs=100, batch_size=50, verbose=2, callbacks=[best_model_High])

# Xây dựng mô hình LSTM
# model_Low = Sequential()
# model_Low.add(LSTM(units=128, return_sequences=True, input_shape = (X_train_Low.shape[1], 1)))
# model_Low.add(LSTM(units=64))
# model_Low.add(Dropout(0.5))
# model_Low.add(Dense(units = 1))
# model_Low.compile(optimizer = 'adam', loss = 'mean_squared_error')

# # Thực hiện huấn luyện mô hình
# save_model_Low = 'Save_train/AI_Low.hdf5'
# best_model_Low = ModelCheckpoint(save_model_Low, monitor='loss', verbose=2, save_best_only=True, mode='auto') # Tìm mô hình huấn luyện tốt nhất
# model_Low.fit(X_train_Low, y_train_Low, epochs=100, batch_size=50, verbose=2, callbacks=[best_model_Low])

y_train = scaler.inverse_transform(y_train_Prev_Close) # Giá thực tế
final_model = load_model('Data/AI_Prev_Close.hdf5')
y_train_predict_Prev_Close = final_model.predict(X_train_Prev_Close)
y_train_predict_Prev_Close = scaler.inverse_transform(y_train_predict_Prev_Close) # Giá dự đoán

value_data_Prev_Close = new_data_prediction_Prev_Close[len_training_new_data-prediction_days:len(new_data_prediction_Prev_Close)-len(test)].values
value_data_Prev_Close = value_data_Prev_Close.reshape(-1, 1)
sc_value = scaler.transform(value_data_Prev_Close)
x_value = []
for i in range(prediction_days, value_data_Prev_Close.shape[0]):
    x_value.append(sc_value[i - prediction_days:i, 0])
x_value = np.array(x_value)

x_value = np.reshape(x_value, (x_value.shape[0], x_value.shape[1], 1))
# print(x_value.shape)

# Dữ liệu value
temp = new_data[len_training_new_data:len(new_data_prediction_Prev_Close)-len(test)]
temp_Prev_Close = temp.filter(['Prev Close'])
y_value_predict = final_model.predict(x_value)
y_value_predict = scaler.inverse_transform(y_value_predict) # Giá dự đoán
train_new_data_Prev_Close = new_data_prediction_Prev_Close[prediction_days:len_training_new_data]
value_new_data_Prev_Close = new_data_prediction_Prev_Close[len_training_new_data:len(new_data_prediction_Prev_Close)-len(test)]

print(value_new_data_Prev_Close.shape)
plt.figure(figsize=(24, 8))
plt.plot(new_data_prediction_Prev_Close["Prev Close"], label = 'Giá thực tế', color = 'red') # Đường giá thực
train_new_data_Prev_Close['Dự đoán'] = y_train_predict_Prev_Close # Thêm dữ liệu

plt.plot(train_new_data_Prev_Close["Dự đoán"], label = 'Giá dự đoán train', color = 'green')
value_new_data_Prev_Close['Dự đoán'] = y_value_predict # Thêm dữ liệu
plt.plot(value_new_data_Prev_Close["Dự đoán"], label = 'Giá dự đoán values', color = 'blue')
plt.title("So sánh giá dự đoán và giá thực tế")
plt.xlabel('Thời gian')
plt.ylabel('Giá đóng cửa (VND)')
plt.legend() # Chú thích
plt.show()



# Đánh giá độ chính xác
print('Độ chính xác của tập train: ', r2_score(y_train, y_train_predict_Prev_Close))
print('Sai số tuyệt đối trung bình tập train: ', mean_absolute_error(y_train, y_train_predict_Prev_Close))
print('Phần trăm sai số tuyệt đối trung bình của tập train: ', mean_absolute_percentage_error(y_train, y_train_predict_Prev_Close), '%')




print('===================================================================')
# Dự đoán giá cổ phiếu của ngài tiếp theo
# print("new_data_prediction_Prev_Close['Prev Close'] = ", new_data_prediction_Prev_Close['Prev Close'])
# print("temp['Prev Close'] = ", temp['Prev Close'])
input = pd.concat((new_data_prediction_Prev_Close['Prev Close'], temp['Prev Close']), axis=0)
# print('input = ', input)
inputs = input[len(input) - len(temp) - prediction_days:].values
# print(inputs.shape)
inputs = inputs.reshape(-1, 1)
inputs = scaler.transform(inputs)
# print(len(inputs))

price_Prev_Close_next_day = [inputs[len(inputs) + 1 - prediction_days:len(inputs + 1), 0]]
price_Prev_Close_next_day = np.array(price_Prev_Close_next_day)
price_Prev_Close_next_day = np.reshape(price_Prev_Close_next_day, (price_Prev_Close_next_day.shape[0], price_Prev_Close_next_day.shape[1], 1))

price_Prev_Close_next_day = final_model.predict(price_Prev_Close_next_day)
prediction = scaler.inverse_transform(price_Prev_Close_next_day)
print("Giá cổ phiếu đóng cửa của ngày tiếp theo là: ", prediction)
Difference = round(float((price_Prev_Close_next_day-inputs[-1])*100), 2)
print('Độ chênh lệch dự đoán là: ', Difference, '%')