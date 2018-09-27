if __name__ == '__main__':
    from keras_model import age_gender_classifier
    test=age_gender_classifier(batch_size=64,lr=0.0001,model_type="one")
    # test.train(epoch=10)
    test.validate(0,0,load_weight=True)
    test.pred_test(0,0)

# from data_stream import data_generator
# test_data=data_generator("age_gender_appa_val",one_hot_gender=False)
#     # ,random_shift=True,random_mirror=True,random_rotate=True,random_scale=True,shuffle=True)
# test_data.show_data()
