from keras_model import age_gender_classifier
test=age_gender_classifier("age_gender_UTK",batch_size=64,lr=0.0001,model_type="one")


# from data_stream import data_generator
# test_data=data_generator("age_gender_UTK"
#     ,random_crop=False,random_mirror=False,random_width=True,random_rotate=True,random_size=True,shuffle=False)
# test_data.show_data()