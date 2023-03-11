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
plt.figure(figsize=(15, 7)) # Xây dựng kích thước khung
plt.title("Biểu đồ giá chứng khoán giá đóng cửa")
plt.plot(data["Date"], data['Prev Close'], label="ACT DATA", color="red")
plt.xlabel("Thời gian")
plt.ylabel("Giá")
plt.grid(color = 'green', linestyle = '--', linewidth = 0.5)
plt.show()

# Cấu trúc lại khung dữ liệu dùng cho dự đoán
new_data = pd.DataFrame(data, columns=['Date', 'Open', 'High', 'Low', 'Prev Close'])
new_data.index = new_data.Date
new_data.drop('Date', axis=1, inplace=True)

# Tạo ra các khung chứa các giá trị dự đoán
# Dự đoán giá 'Prev Close' (Giá trước đóng cửa)
new_data_prediction_Prev_Close = new_data.filter(['Prev Close'])

# Chia dữu liệu thành 3 tập: Dữ liệu train, dữ liệu values và dữ liệu test.
# Trong đó: 80% làm dữ liệu train, còn lại được chia ra tiếp tục cho dữ liệu values và dữ liệu test
# print("Độ dài ban đầu: ", len(new_data))
len_training_new_data = math.ceil(len(new_data)* .8)
# print("Độ dài sau khi sử lý lấy dữ liệu train: ", len_training_new_data)
# print("Kết quả dữ liệu tập train: ", len_training_new_data/len(new_data) * 100, "%")
train = new_data[:len_training_new_data]
# Số lượng dữ liệu ban đầu là 5075, dùng 80% (nghĩa là sử dụng 4060) dữ liệu cho train
# Còn lại: 5075 - 4060 = 1015 (Số lượng dữ liệu này được tiếp tục chia làm dữ liệu value và dữ liệu test)
len_values_test = len(new_data) - len(train)
print("Độ dài dữ liệu còn lại là: ", len_values_test)
# Lấy 80% dữ liệu còn lại làm dữ liệu values
len_training_new_values = math.ceil(len_values_test* .8)
# print("Độ dài sau khi sử lý lấy dữ liệu values: ", len_training_new_values)
# print("Kết quả dữ liệu tập values: ", (len_training_new_values/len_values_test) * 100, "%")
temp = new_data[len_training_new_data:]
value = temp[:len_training_new_values]
test = temp[len_training_new_values:]
# print('=====================================')
# print("Đây là dành cho tệp train\n", train)
# print("Đây là dành cho tệp value\n", value)
# print("Đây là dành cho tệp test\n", test)
# print('------------------------------Kết quả------------------------------')
# print('Số lượng dành cho tập train: ', len_training_new_data)
# print('Số lượng dành cho tập values: ', len(value))
# print('Số lượng dành cho tập test: ', len(test))
# print('===================================================================')


# Tăng tốc độ xử lý của mô hình bằng cách chuẩn hóa lại dữ liệu
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data_Prev_Close = scaler.fit_transform(train['Prev Close'].values.reshape(-1,1))
print('Độ dài của "Prev Close" là  : ', len(scaled_data_Prev_Close))

# Tạo X_train và y_train ==> Mục tiêu: Dựa vào khoảng thời gian trước đó 60 ngày để có thể dự đoán được giá của 60 ngày tiếp theo
prediction_days = 60
X_train_Prev_Close, y_train_Prev_Close = [], []
for x in range(prediction_days, len(scaled_data_Prev_Close)):
  X_train_Prev_Close.append(scaled_data_Prev_Close[x-prediction_days:x, 0])
  y_train_Prev_Close.append(scaled_data_Prev_Close[x, 0])

X_train_Prev_Close, y_train_Prev_Close = np.array(X_train_Prev_Close), np.array(y_train_Prev_Close) # Xếp dữ liệu thành một mảng
X_train_Prev_Close = np.reshape(X_train_Prev_Close, (X_train_Prev_Close.shape[0], X_train_Prev_Close.shape[1], 1))
y_train_Prev_Close = np.reshape(y_train_Prev_Close, (y_train_Prev_Close.shape[0], 1))
print('Đây là dành cho train Prev Close: ', X_train_Prev_Close.shape)
# Xây dựng mô hình LSTM
model_Prev_Close = Sequential()
model_Prev_Close.add(LSTM(units=128, return_sequences=True, input_shape = (X_train_Prev_Close.shape[1], 1)))
model_Prev_Close.add(LSTM(units=64))
model_Prev_Close.add(Dropout(0.5))
model_Prev_Close.add(Dense(units = 1))
model_Prev_Close.compile(optimizer = 'adam', loss = 'mean_squared_error')
print(model_Prev_Close.summary())
# Thực hiện huấn luyện mô hình
epochs = 20
save_model_Prev_Close = 'Data/AI_NEW.hdf5'
best_model_Prev_Close = ModelCheckpoint(save_model_Prev_Close, monitor='loss', verbose=2, save_best_only=True, mode='auto') # Tìm mô hình huấn luyện tốt nhất
history = model_Prev_Close.fit(X_train_Prev_Close, y_train_Prev_Close, epochs=epochs, batch_size=50, verbose=2, callbacks=[best_model_Prev_Close])

# list all data in history
print(history.history.keys())

plt.figure(1)
plt.plot(history.history['loss'], label='Loss')
plt.title('Biểu đồ trực quan hóa sự mất mát trong quá trình huấn luyện')
plt.legend()
plt.show()

y_train = scaler.inverse_transform(y_train_Prev_Close) # Giá thực tế
final_model = load_model('Data/AI_NEW.hdf5')
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

# input = temp['Close']
# inputs = input[len(input) - len(temp) - prediction_days:].values
# inputs = inputs.reshape(-1, 1)
# inputs = scaler.transform(inputs)

# price_Prev_Close_next_day = [inputs[len(inputs) + 1 - prediction_days:len(inputs + 1), 0]]
# price_Prev_Close_next_day = np.array(price_Prev_Close_next_day)
# price_Prev_Close_next_day = np.reshape(price_Prev_Close_next_day, (price_Prev_Close_next_day.shape[0], price_Prev_Close_next_day.shape[1], 1))

# price_Prev_Close_next_day = final_model.predict(price_Prev_Close_next_day)
# prediction = scaler.inverse_transform(price_Prev_Close_next_day)
# print("Giá cổ phiếu đóng cửa của ngày tiếp theo là: ", prediction)
# Difference = round(float((price_Prev_Close_next_day-inputs[-1])*100), 2)
# print('Độ chênh lệch dự đoán là: ', Difference, '%')

print(train_new_data_Prev_Close)
print(value_new_data_Prev_Close)
Data_DATE_FILE_TRAIN = data[prediction_days:len_training_new_data]
Data_DATE_FILE_VALUE = data[len_training_new_data:len(new_data_prediction_Prev_Close)-len(test)]

Data_DATE_FILE_TRAIN = Data_DATE_FILE_TRAIN.filter(['Date'])
Data_DATE_FILE_VALUE = Data_DATE_FILE_VALUE.filter(['Date'])

# print(Data_DATE_FILE_TRAIN)
# print(Data_DATE_FILE_VALUE)
# Data_DATE_FILE_TRAIN = scaler.inverse_transform(Data_DATE_FILE_TRAIN)
# Đưa ra bảng Excel ==> Kiểm tra xem đã tồn tại file hay chưa, nếu chưa tồn tại thì lưu, nếu đã tồn tại thì xóa file cũ và lưu thành file mới
import os
files_TRAIN_EXCEL = 'Data/File_Data_Train.xlsx'
if os.path.exists(files_TRAIN_EXCEL):
  os.remove(files_TRAIN_EXCEL)
train_new_data_Prev_Close = train_new_data_Prev_Close.merge(Data_DATE_FILE_TRAIN, on='Date')
train_new_data_Prev_Close.to_excel(files_TRAIN_EXCEL, index=False)

files_VALUE_EXCEL = 'Data/File_Data_Value.xlsx'
if os.path.exists(files_VALUE_EXCEL):
  os.remove(files_VALUE_EXCEL)
value_new_data_Prev_Close = value_new_data_Prev_Close.merge(Data_DATE_FILE_VALUE, on='Date')
value_new_data_Prev_Close.to_excel(files_VALUE_EXCEL, index=False)