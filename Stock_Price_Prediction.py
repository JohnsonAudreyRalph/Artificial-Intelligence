import pandas as pd # Đọc dữ liệu từ tập dữ liệu
import matplotlib.pyplot as plt # Vẽ các biểu đồ đường dữ liệu
import numpy as np # Xử lý dữ liệu
import math
from sklearn.preprocessing import MinMaxScaler # Chuẩn hóa dữ liệu
from sklearn.metrics import r2_score, mean_absolute_error, mean_absolute_percentage_error # Đo mức độ phù hợp, sai số tuyệt đối trung bình và phần trăm sau số tuyệt đối trung bình
from keras.models import load_model, Sequential # Tải mô hình, tạo một INSTANCE của mô hình Neural network
from keras.callbacks import ModelCheckpoint # Lưu lại mô hình huấn luyện tốt nhất
from keras.layers import LSTM, Dense, Dropout

# Bước 2: Lấy dữ liệu và thực hiện mô tả dữ liệu
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
plt.plot(data["Date"], data['Prev Close'], label="Giá đóng cửa", color="red")
plt.xlabel("Thời gian")
plt.ylabel("Giá")
plt.grid(color = 'green', linestyle = '--', linewidth = 0.5)
plt.show()

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
# y_train_Prev_Close = np.reshape(y_train_Prev_Close, (y_train_Prev_Close.shape[0], 1))

# Xây dựng mô hình LSTM
model_Prev_Close = Sequential()
model_Prev_Close.add(LSTM(units=128, return_sequences=True, input_shape = (X_train_Prev_Close.shape[1], 1)))
model_Prev_Close.add(LSTM(units=64))
model_Prev_Close.add(Dropout(0.5))
model_Prev_Close.add(Dense(units = 1))
model_Prev_Close.compile(optimizer = 'adam', loss = 'mean_squared_error')

# # Thực hiện huấn luyện mô hình
# save_model_Prev_Close = 'Data/AI_Prev_Close.hdf5'
# best_model_Prev_Close = ModelCheckpoint(save_model_Prev_Close, monitor='loss', verbose=2, save_best_only=True, mode='auto') # Tìm mô hình huấn luyện tốt nhất
# model_Prev_Close.fit(X_train_Prev_Close, y_train_Prev_Close, epochs=100, batch_size=50, verbose=2, callbacks=[best_model_Prev_Close])

# Dữ liệu train <==> Với y_train: là giá thực và y_train_predict: là giá dự đoán
y_train_Prev_Close = value['Prev Close'].values # Thực tế
Total_dataset_Prev_Close = pd.concat((new_data_prediction_Prev_Close['Prev Close'], value['Prev Close']), axis = 0)
inputs_Prev_Close = Total_dataset_Prev_Close[len(Total_dataset_Prev_Close) - len(value) - prediction_days:].values
inputs_Prev_Close = inputs_Prev_Close.reshape(-1, 1)
inputs_Prev_Close = scaler.transform(inputs_Prev_Close)
X_value = []
for x in range(prediction_days, len(inputs_Prev_Close)):
  X_value.append(inputs_Prev_Close[x-prediction_days:x, 0])
X_value = np.array(X_value)
print(X_value.shape)
X_value = np.reshape(X_value, (X_value.shape[0], X_value.shape[1], 1))

model = load_model('Data/AI_Prev_Close.hdf5')
y_train_predict_Prev_Close = model.predict(X_value)
y_train_predict_Prev_Close = scaler.inverse_transform(y_train_predict_Prev_Close)








plt.figure(figsize=(15, 7)) # Xây dựng kích thước khung
plt.title('Biểu đồ so sánh giữa thực tế và dự đoán')
plt.plot(y_train_Prev_Close, label="Thực tế", color="gray")
plt.plot(y_train_predict_Prev_Close, label="Dự đoán", color="red")
plt.xlabel("Thời gian")
plt.ylabel("Giá")
plt.grid(color = 'green', linestyle = '--', linewidth = 0.5)
plt.legend()
plt.show()