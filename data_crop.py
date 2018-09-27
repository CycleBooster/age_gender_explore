import dlib
import cv2
import os
import numpy as np
import time
from progress.bar import Bar
def paste_face(img,face_coor,paste_size):
    img_h,img_w=img.shape[:2]
    left,right,top,bottom=face_coor
    x_center=(int)((left+right)/2)
    face_h=bottom-top
    crop_y_start=(int)(top-0.4*face_h)
    crop_y_end=  (int)(bottom+0.1*face_h)
    crop_x_start=(int)(x_center-0.5*(crop_y_end-crop_y_start))
    crop_x_end=  (int)(x_center+0.5*(crop_y_end-crop_y_start))
    target_x_start=0
    target_x_end=crop_x_end-crop_x_start
    target_y_start=0
    target_y_end=crop_y_end-crop_y_start
    background=np.zeros((target_y_end,target_x_end,3),dtype=np.uint8)
    if crop_x_start<0:
        target_x_start=(-crop_x_start)
        crop_x_start=0
    if crop_y_start<0:
        target_y_start=(-crop_y_start)
        crop_y_start=0
    if crop_x_end>=img_w:
        target_x_end+=img_w-crop_x_end-1
        crop_x_end=img_w-1
    if crop_y_end>=img_h:
        target_y_end+=img_h-crop_y_end-1
        crop_y_end=img_h-1
    background[target_y_start:target_y_end,target_x_start:target_x_end]=img[crop_y_start:crop_y_end,crop_x_start:crop_x_end]
    out_face=cv2.resize(background,paste_size)
    return out_face
def check_name(img_name):
    check_split=img_name.split(".")
    if check_split[-1]!="jpg":
        return False
    if check_split[-2]=="jpg_face":
        return False
    return True
def draw_face(img,face_rects,scores):
    for i, d in enumerate(face_rects):
        x1 = d.left()
        y1 = d.top()
        x2 = d.right()
        y2 = d.bottom()
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 4, cv2.LINE_AA)
        cv2.putText(img, str(scores[i]), (x1, y1), cv2.FONT_HERSHEY_DUPLEX,
          0.7, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imshow("draw face", img)
    cv2.waitKey(0)
def crop_face(data_path):
    save_path="./test_photo/crop/"
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    img_name_list=os.listdir(data_path)
    data_len=len(img_name_list)
    bar = Bar('Processing', max=data_len,fill='-')
    for img_name in img_name_list:
        bar.next()
        time_list=[]
        time_list.append(time.time())
        img_path=data_path+img_name
        img=cv2.imread(img_path,1)
        time_list.append(time.time())
        if not check_name(img_name):
            continue
        
        detector = dlib.get_frontal_face_detector()
        # face_rects = detector(img, 0)
        face_rects, scores, idx = detector.run(img, 0, -0.1)
        time_list.append(time.time())
        if len(scores)!=1:
            # print(img_name)
            # print(len(scores))
            # draw_face(img,face_rects,scores)
            continue
        for i, d in enumerate(face_rects):
            face_img=paste_face(img,[d.left(),d.right(),d.top(),d.bottom()],(128,128))
            
            time_list.append(time.time())
            # for i in range(len(time_list)-1):
            #     print(time_list[i+1]-time_list[i])
            # cv2.imshow("Face Detection", face_img)
            # cv2.waitKey(0)
            try:
                cv2.imwrite(save_path+img_name,face_img)
            except:
                print()
                print(img_name)
    bar.finish()
def try_read(data_path):
    img_name_list=os.listdir(data_path)
    data_len=len(img_name_list)
    bar = Bar('Processing', max=data_len,fill='-')
    for img_name in img_name_list:
        bar.next()
        img_path=data_path+img_name
        img=cv2.imread(img_path,1)
    bar.finish()
def find_lost(origin_path,target_path):#only work in used crop_face in the same PC
    origin_name_list=os.listdir(origin_path)
    target_name_list=os.listdir(target_path)
    origin_index=0
    for target_name in target_name_list:
        while origin_name_list[origin_index]!=target_name:
            print(origin_name_list[origin_index])
            origin_index+=1
        origin_index+=1
if __name__ == '__main__':
    crop_face(data_path=".\\data\\appa-real\\test\\")
    # try_read(data_path=".\\test_photo\\crop\\")
    # find_lost(".\\data\\part3\\",".\\test_photo\\crop_part3\\")