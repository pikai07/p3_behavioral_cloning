import json
from helper import *
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Lambda
from keras.layers import Convolution2D, MaxPooling2D, ELU, Flatten
from keras.optimizers import Adam

#=====================================
# Using Lenet Architecture
#=====================================
def LeNet_model():
    # model paramaters
    input_shape = (64, 64, 1)
    filter_size = 3 # extract finer features with a 3x3 filter
    pool_size = (4,4)
    
    # Create model
    model = Sequential()
    
    # Normalize the data
    model.add(Lambda(lambda x: x/255.-0.5, input_shape=input_shape))
    
    # Convolution 1 -> activation -> pooling
    model.add(Convolution2D(6, filter_size, filter_size, init='he_normal', border_mode='valid'))
    model.add(ELU())
    model.add(MaxPooling2D(pool_size=pool_size))
    
    # Convolution 2 -> activation -> pooling
    model.add(Convolution2D(16, filter_size, filter_size, init='he_normal', border_mode='valid'))
    model.add(ELU())
    model.add(MaxPooling2D(pool_size=pool_size))
    
    # Flatten
    model.add(Flatten())
    
    # Fully connected layers
    model.add(Dense(120, init='he_normal'))      # fc1
    model.add(ELU())
    model.add(Dense(84, init='he_normal'))       # fc2
    model.add(ELU())
    model.add(Dense(1, init='he_normal'))        # output
    return model

def save_model(model, json_file, weights_file):
    # Save model architecture
    with open(json_file,'w' ) as f:
        json.dump(model.to_json(), f)
    
    # Save model weights
    model.save_weights(weights_file)


if __name__ == "__main__":
    col_size = 64
    row_size = 64
    
    #=========================================
    # Preorocess data
    #=========================================
    print("Preprocessing Data....")
    print(" ")
    
    # Load data
    full_data_log = load_data()
    
    
    # Preprocess images and load train and test features into numpy arrays
    train_features, train_labels = load_training_data(full_data_log, row_size, col_size)
    
    # Create validation set with a 90-10 split
    train_features, train_labels = shuffle(train_features, train_labels)
    train_features, valid_features, train_labels, valid_labels = train_test_split(
                                                                              train_features,
                                                                              train_labels,
                                                                              test_size=0.10
                                                                            random_state=832289)
    #=============================================
    # Train and Save the Neural Network Parameters
    #=============================================
    print("Training Model....")
    print(" ")
    model = LeNet_model()

    # Compile model using mean square error as the loss parameter
    # Optimizer - Adam
    model.compile(optimizer='adam', loss='mse', metric=['accuracy'])

    # Train model for 10 Epochs
    history = model.fit(train_features, train_labels,
                   batch_size=128, nb_epoch=10, shuffle=True,
                   validation_data=(valid_features, valid_labels))

    # Save model architecture and weights
    save_model(model, 'model.json', 'model.h5')
    print(" ")
    print("Model saved....")