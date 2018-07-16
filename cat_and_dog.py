from keras.utils import np_utils  
from keras.models import load_model
from sklearn.model_selection import train_test_split
import numpy as np
import csv, os, cv2, random, glob, h5py, keras
import matplotlib.pyplot as plt
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
np.set_printoptions(suppress=True)
# 參數設定------------------------------------------------------------------------------------
map_characters = {0: 'cat', 1: 'dog'}
img_width = 64 
img_height = 64
num_classes = len(map_characters) # 要辨識的種類
test_size = 0.1
train_imgsPath = "train"
#--------------------------------------------------------------------------------------------

# 將訓練資料圖像從檔案系統中取出並進行
def load_pictures():
    pics = []
    labels = []
    
    for k, v in map_characters.items(): # k: 數字編碼 v: label
        # 把所有圖像檔的路徑捉出來
        pictures = [k for k in glob.glob(train_imgsPath + "/" + v + "/*")]       
        print(v + " : " + str(len(pictures))) # 看一下各有多少訓練圖像
        for i, pic in enumerate(pictures):
            #print(pic)
            tmp_img = cv2.imread(pic)
            #cv2.namedWindow("Image")
            #cv2.imshow("Image", tmp_img)
            #cv2.waitKey(0)
            # 由於OpenCv讀圖像時是以BGR (Blue-Green-Red), 我們把它轉置成RGB (Red-Green-Blue)
            tmp_img = cv2.cvtColor(tmp_img, cv2.COLOR_BGR2RGB)
            tmp_img = cv2.resize(tmp_img, (img_height, img_width)) # 進行resize            
            pics.append(tmp_img) # 塞到 array 裡
            labels.append(k)    
    return np.array(pics), np.array(labels)

# 取得訓練資料集與驗證資料集
def get_dataset(save=False, load=False):
    if load: 
        # 從檔案系統中載入之前處理保存的訓練資料集與驗證資料集
        h5f = h5py.File('dataset.h5','r')
        X_train = h5f['X_train'][:]
        x_val = h5f['x_val'][:]
        h5f.close()
        
        # 從檔案系統中載入之前處理保存的訓練資料標籤與驗證資料集籤
        h5f = h5py.File('labels.h5', 'r')
        y_train = h5f['y_train'][:]
        y_val = h5f['y_val'][:]
        h5f.close()
    else:
        # 從最原始的圖像檔案開始處理
        X, y = load_pictures()
        #y = keras.utils.to_categorical(y, num_classes) # 這裡只有分兩類
        
        # 將整個資料集切分為訓練資料集與驗證資料集 (90% vs. 10%)
        X_train, x_val, y_train, y_val = train_test_split(X, y, test_size=test_size) 
        if save: # 保存尚未進行歸一化的圖像數據
            h5f = h5py.File('dataset.h5', 'w')
            h5f.create_dataset('X_train', data=X_train)
            h5f.create_dataset('x_val', data=x_val)
            h5f.close()
            
            h5f = h5py.File('labels.h5', 'w')
            h5f.create_dataset('y_train', data=y_train)
            h5f.create_dataset('y_val', data=y_val)
            h5f.close()
    
    # 進行圖像每個像素值的型別轉換與normalize成零到一
    X_train = X_train.astype('float32') / 255.
    x_val = x_val.astype('float32') / 255.
    print("Train", X_train.shape, y_train.shape)
    print("Test", x_val.shape, y_val.shape)
    
    return X_train, x_val, y_train, y_val

# 取得訓練資料集與驗證資料集  
x_train, x_val, y_train, y_val = get_dataset()

from keras.models import Sequential  
from keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPool2D,Activation,BatchNormalization

model = Sequential()

model.add(Conv2D(32, kernel_size=(5, 5), input_shape=(64, 64, 3), activation='relu', padding='same'))
model.add(MaxPool2D())
model.add(Dropout(0.3))

model.add(Conv2D(64, kernel_size=(5, 5), activation='relu', padding='same'))
model.add(MaxPool2D())
model.add(Dropout(0.3))

model.add(Conv2D(128, kernel_size=(5, 5), activation='relu', padding='same'))
model.add(MaxPool2D())
model.add(Dropout(0.5))

model.add(Conv2D(256, kernel_size=(5, 5), activation='relu', padding='same'))
model.add(MaxPool2D())
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(1, activation='sigmoid'))

model.summary()

# 定義訓練方式  
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])  

# 開始訓練  
train_history = model.fit(x=x_train, y=y_train,
                          validation_data=(x_val,y_val),
                          epochs=400, 
                          batch_size=100,
                          shuffle=True,
                          callbacks=[ModelCheckpoint('model.h5', save_best_only=True)], # 儲存最好的 model 來做 testing
                          verbose=2)

def plot_train_history(history, train_metrics, val_metrics):
    plt.plot(history.history.get(train_metrics),'-o')                              
    plt.plot(history.history.get(val_metrics),'-o')
    plt.ylabel(train_metrics)
    plt.xlabel('Epochs')
    plt.legend(['train', 'validation'])

def show_train_and_val_result():
    # 透過趨勢圖來觀察訓練與驗證的走向 (特別去觀察是否有"過擬合(overfitting)"的現象)
    import matplotlib.pyplot as plt   
    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plot_train_history(train_history, 'loss','val_loss')

    plt.subplot(1,2,2)
    plot_train_history(train_history, 'acc','val_acc')

    plt.show()

show_train_and_val_result()

model = load_model('model.h5')
# testing
test_imgsPath = 'test'
test_imgs = os.listdir(test_imgsPath)
def load_test_imgs():
    pics = []
    labels = []
    for img in test_imgs:
        label = int(img[:-4])
        #print(type(label))
        pic_path = os.path.join(test_imgsPath,img)
        tmp = cv2.imread(pic_path)
        #cv2.namedWindow("Image")
        #cv2.imshow("Image", tmp)
        #cv2.waitKey(0)
        # 由於OpenCv讀圖像時是以BGR (Blue-Green-Red), 我們把它轉置成RGB (Red-Green-Blue)
        tmp = cv2.cvtColor(tmp, cv2.COLOR_BGR2RGB)
        tmp = cv2.resize(tmp, (img_height, img_width)) # 進行resize            
        pics.append(tmp) # 塞到 array 裡
        labels.append(label)    
    
    return np.array(pics), np.array(labels)

x_test, label_x = load_test_imgs()
x_test = x_test.astype('float32') / 255.

y_pred = model.predict(x_test)
#y_pred = y_pred.clip(min=0.005, max=0.995)
print(y_pred[:10])

#整理output csv file
array = []
for i in range(len(y_pred)):
    tmp = {"y_pred":y_pred[i].tolist(),"label":label_x[i]}
    #print(tmp)
    array.append(tmp)

array = sorted(array, key=lambda array:array["label"])
#print(array)

def saveResult(result):
    with open ('result.csv', mode='w',newline="\n") as write_file:
        writer = csv.writer(write_file)
        writer.writerow(["id","label"])
        for i in range(len(result)):
            writer.writerow([result[i]["label"], result[i]["y_pred"][0]])

saveResult(array)
print("original structure without clip")