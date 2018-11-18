import os 
import numpy as np 
import pandas as pd
from keras.preprocessing import image as kimage
from keras.applications.vgg16 import preprocess_input
    
PATH = os.getcwd()
#path of test and train folders are located a directory above
TEST_PATH = os.path.join(os.path.split(PATH)[0], 'test')
TEST_BATCH = os.listdir(os.path.join(TEST_PATH, 'test_images'))



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
    # x_test = []
    # for sample in TEST_BATCH:
    #     x_test.append(sample[:-4])
        
    # # creating dataframe to be used
    # df = pd.DataFrame(x_test, columns=[''])

    return (df, test_generator)

def create_csv():
    """create csv from predicted dataframe"""
    pass

def model():
    """replicate model by loading architecture and weights"""
    pass

def predict():
    """output predicted labels for test"""
    pass

if __name__=="__main__":
    load_test_images()