# from keras_model import age_gender_classifier
# test=age_gender_classifier("none")



from data_stream import data_generator
test_data=data_generator("age_gender_appa")
test_data.show_data()