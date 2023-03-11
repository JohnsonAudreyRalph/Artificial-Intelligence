from django.shortcuts import render, redirect
from django.core.files.storage import FileSystemStorage
from django.contrib import messages
import os
import pandas as pd
from django.core.files.storage import default_storage

from django.conf import settings
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from keras.models import load_model

# Create your views here.
def Home(request):
    global attribute
    if(request.method == 'POST'):
        Upload_File = request.FILES['document']
        attribute = request.POST.get('attributeid')
        # print(attributeid)
        if (Upload_File.name.endswith('.csv')):
            saveFile = FileSystemStorage()
            print("==============Đây là tên file", Upload_File)
            file_path = os.path.join(settings.MEDIA_ROOT, str(Upload_File))
            if os.path.exists(file_path):
                os.remove(file_path)
                print('Đã xóa file')
            else:
                print('File không tồn tại')
            print(f"Đã lưu file: {Upload_File}")
            name = saveFile.save(Upload_File.name, Upload_File)
            cmd = os.getcwd()
            File_Directory = cmd + '\media\\' + name
            # print(File_Directory)
            readFile(File_Directory)
            request.session['attribute'] = attribute
            if attribute not in data.axes[1]:
                messages.warning(request, 'Hãy nhập đúng tên cột mà bạn muốn tính')
            else:
                # print(attribute)
                return redirect(Results)
        else:
            messages.warning(request, 'Không thể thực hiện Upload! Hãy kiểm tra lại file')
    return render(request, 'index.html')


def readFile(fileName):
    global rows, columns, data, file, missing_values
    file = pd.read_csv(fileName, sep='[:;,|_]', engine='python')
    data = pd.DataFrame(data=file, index=None)
    # print(data)
    rows = len(data.axes[0])
    columns = len(data.axes[1])
    missing_values = ['0', '?', '--']
    null_data = data[data.isnull().any(axis=1)]
    missing_values = len(null_data)

def Price_Prediction():
    global Prediction, Difference
    scaler = MinMaxScaler(feature_range=(0, 1))
    print("==========================================")
    print(data[attribute])
    input = data[attribute]
    inputs = input.values
    # print('1', inputs)
    inputs = inputs.reshape(-1, 1)
    # print('2', inputs)
    scaler.fit(inputs)
    inputs = scaler.transform(inputs)
    # print('3', len(inputs))
    # print('4', len(data))
    final_model = load_model('Data/AI_Prev_Close.hdf5')
    # final_model = load_model('AI.hdf5')
    Price_next_day = [inputs[len(inputs + 1) - 60:len(inputs + 1), 0]]
    # print('len(inputs + 1) = ', len(inputs + 1))
    # print('len(inputs + 1) - 60 = ', len(inputs + 1) - 60)
    Price_next_day = np.array(Price_next_day)
    Price_next_day = np.reshape(Price_next_day, (Price_next_day.shape[0], Price_next_day.shape[1], 1))
    Price_next_day = final_model.predict(Price_next_day)
    Prediction = scaler.inverse_transform(Price_next_day)
    Prediction = Prediction[0][0]
    print("Giá cổ phiếu đóng cửa của ngày tiếp theo là: ", Prediction)
    Difference = round(float((Price_next_day-inputs[-1])*100), 2)
    Difference = abs(Difference)
    print('Độ chênh lệch dự đoán là: ', Difference, '%')
    print("==========================================")

def Results(request):
    message = 'Kết quả nhận thấy: File này có: ' + str(rows) + ' hàng, ' + str(columns) + ' cột. \nSố phần tử trống là: ' + str(missing_values)
    messages.warning(request, message)
    dashboard = []
    DAY = []
    for att in data[attribute]:
        dashboard.append(att)
    for day in data['Date']:
        DAY.append(day)
    
    # print('===========================================')
    # print(dashboard) # Dữ liệu về giá của loại nhập vào từ cái trước
    # print(DAY)
    # print(attribute)
    # print('===========================================')
    # print('Đây là listKey: \n', DAY)
    # print('Đây là listvalues: \n', dashboard)
    Price_Prediction()
    context = {
        'listKeys': DAY,
        'listvalues': dashboard,
        'attribute':attribute,
        'Prediction':Prediction,
        'Difference': Difference
    }
    return render(request, 'Chart.html', context)
