import os 
import numpy as np 
import pandas as pd
from keras.preprocessing import image as kimage
from keras.models import load_model
    
PATH = os.getcwd()
#path of test and train folders are located a directory above
TEST_PATH = os.path.join(os.path.split(PATH)[0], 'test')
TEST_BATCH = os.listdir(os.path.join(TEST_PATH, 'test_images'))
MODEL = os.path.join(PATH, 'fashion_mnist_trained.h5')


def load_test_images():
    """loads test images from the folder"""
    
    test_datagen = kimage.ImageDataGenerator()
    test_generator = test_datagen.flow_from_directory(
    directory=TEST_PATH,
    color_mode="grayscale",
    target_size=(28, 28),
    batch_size=1,
    shuffle=False
        )
    x_test = []
    for sample in TEST_BATCH:
        # remove '.jpg'
        x_test.append(sample[:-4])
        
    # creating dataframe to be used
    df = pd.DataFrame(x_test, columns=['ImageID'])

    return (df, test_generator)

def create_csv(initial_df, classes):
    """create csv from predicted dataframe
    
    initial_df: dataframe containing 'ImageCategory' only
    classes: list of yhat predictions
    
    returns:
        location: full dataframe"""
    
    classes = pd.Series(classes)
    initial_df['Category'] = classes.values
    initial_df.to_csv('predictions.csv')
    return initial_df

def model():
    """replicate model by loading architecture and weights"""
    model = load_model(MODEL)
    return model

def predict(model, test_data):
    """output predicted labels for test
    
    model: keras model
    test_data: numerical stream of images to predict

    returns:
        classes: y hat predictions
    """
    classes = model.predict_classes(test_data) 
    return classes

if __name__=="__main__":
    load_test_images()