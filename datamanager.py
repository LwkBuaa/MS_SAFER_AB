# create data and label for CK+
# 0=anger,1=disgust,2=fear,3=happy,4=sadness,5=surprise,6=contempt
# contain 135,177,75,207,84,249,54 images
import csv
import os
import numpy as np
import cv2
from PIL import Image


def CKManager():
    ck_path = r'DataSet\CK+'
    Emotion_path = os.path.join(ck_path, "Emotion")
    ck_img_path = os.path.join(ck_path, "cohn-kanade-images")
    anger_path = os.path.join(ck_path, "emotion_cls", 'anger')
    disgust_path = os.path.join(ck_path, "emotion_cls", 'contempt')
    fear_path = os.path.join(ck_path, "emotion_cls", 'disgust')
    happy_path = os.path.join(ck_path, "emotion_cls", 'fear')
    sadness_path = os.path.join(ck_path, "emotion_cls", 'happy')
    surprise_path = os.path.join(ck_path, "emotion_cls", 'sadness')
    contempt_path = os.path.join(ck_path, "emotion_cls", 'surprise')
    all_emotion_cls = ['anger', 'contempt', 'disgust', 'fear', 'happy', 'sadness', 'surprise']
    with open(ck_path + "/emotion_cls" + "/labels.txt", "w+") as f:
        for id, emotion_cls in enumerate(all_emotion_cls):
            data = str(id+1) + "\t" + emotion_cls + "\n"
            f.write(data)
        f.close()
    all_emotion_path = [anger_path, disgust_path, fear_path, happy_path, sadness_path, surprise_path, contempt_path]

    for id, emotion_path in enumerate(all_emotion_path):
        print(id, emotion_path)
        if not os.path.exists(emotion_path):
            os.makedirs(emotion_path)

    img_list = []
    for people in os.listdir(Emotion_path):
        for emotion in os.listdir(os.path.join(Emotion_path, people)):
            for emotion_txt in os.listdir(os.path.join(Emotion_path, people, emotion)):
                with open(os.path.join(Emotion_path, people, emotion, emotion_txt)) as f:
                    data = f.readline()
                    img_cls = data.strip()[0]
                img_id = people + '_' + emotion + '_' + emotion_txt.split('_')[2] + '.png'
                img_list.append((os.path.join(ck_img_path, people, emotion, img_id), img_cls))
    with open(ck_path + "/emotion_cls" + "/train_test.txt", "w+") as f:
        for img_path, img_cls_num in img_list:
            print(img_path, img_cls_num)
            img = Image.open(img_path)
            img_cls = all_emotion_cls[int(img_cls_num)-1]
            datapath = all_emotion_path[int(img_cls_num)-1]
            print(img_cls_num, img_cls, datapath)
            print(os.path.join(datapath, img_path.split('\\')[-1]))
            img.save(os.path.join(datapath, img_path.split('\\')[-1]))
            data = img_cls + '\\' + img_path.split('\\')[-1] + '\n'
            f.write(data)
    f.close()

    print('Save data finish!!!')


def FERPlusManager():
    ferplus_path = "DataSet\FERPlus\data"
    Test_path = os.path.join(ferplus_path, 'FER2013Test')
    Train_path = os.path.join(ferplus_path, 'FER2013Train')
    Valid_path = os.path.join(ferplus_path, 'FER2013Valid')
    test_train_valid_path = [Test_path, Train_path, Valid_path]
    Test_label_path = os.path.join(ferplus_path, 'test.txt')
    Train_label_path = os.path.join(ferplus_path, 'train.txt')
    Valid_label_path = os.path.join(ferplus_path, 'valid.txt')
    test_train_valid_label_path = [Test_label_path, Train_label_path, Valid_label_path]
    anger_path = os.path.join(ferplus_path, "emotion_cls", 'anger')
    disgust_path = os.path.join(ferplus_path, "emotion_cls", 'disgust')
    fear_path = os.path.join(ferplus_path, "emotion_cls", 'fear')
    happy_path = os.path.join(ferplus_path, "emotion_cls", 'happy')
    sadness_path = os.path.join(ferplus_path, "emotion_cls", 'sadness')
    surprise_path = os.path.join(ferplus_path, "emotion_cls", 'surprise')
    neutral_path = os.path.join(ferplus_path, "emotion_cls", 'neutral')
    all_emotion_cls = ['anger', 'disgust', 'fear', 'happy', 'sadness', 'surprise', 'neutral']
    all_emotion_path = [anger_path, disgust_path, fear_path, happy_path, sadness_path, surprise_path, neutral_path]
    for id, emotion_path in enumerate(all_emotion_path):
        print(id, emotion_path)
        if not os.path.exists(emotion_path):
            os.makedirs(emotion_path)
    for file in test_train_valid_label_path:
        with open(file, "r") as f:
            while True:
                data = f.readline()
                if data:
                    img_cls, img_path = data.split()
                    img = Image.open(os.path.join(ferplus_path, img_path))
                    img.save(os.path.join(all_emotion_path[int(img_cls)], img_path.split('\\')[-1]))
                else:
                    break


if __name__ == '__main__':
    FERPlusManager()
