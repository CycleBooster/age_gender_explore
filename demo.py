# from keras_model import age_gender_classifier
# test=age_gender_classifier("none")
# from data_stream import data_generator
# test_data=data_generator("age_gender_UTK"
#     ,random_crop=False,random_mirror=False,random_width=True,random_rotate=True,random_size=True,shuffle=False)
# test_data.show_data().
import cv2
img=cv2.imread("test_photo/African_Bush_Elephant.jpg")
img=cv2.resize(img,(200,200))
cv2.imshow("test view",img)
cv2.waitKey()