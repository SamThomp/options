from OP_Predict_Config import *

def split_target_data(train_data, test_data, column_stop): 

    # Separate the target from the parameters for the training data
    x_train = train_data[[i for i in range(1, column_stop )]].values
    y_train = train_data[0]

    # Separate the target from the parameters for the test data
    x_test = test_data[[i for i in range(1,  column_stop )]].values
    y_test = test_data[0]

    sequence_length = int(PAST / STEP)

    return x_train, y_train, x_test, y_test, sequence_length


def get_keras_datasets(x_train, y_train, x_test, y_test, sequence_length ):
    dataset_train = keras.preprocessing.timeseries_dataset_from_array(
        x_train,
        y_train,
        sequence_length=sequence_length,
        sampling_rate=STEP,
        batch_size=BATCH_SIZE,
    )

    dataset_val = keras.preprocessing.timeseries_dataset_from_array(
        x_test,
        y_test,
        sequence_length=sequence_length,
        sampling_rate=STEP,
        batch_size=BATCH_SIZE,
    )    

    return dataset_train, dataset_val

"""
## Training
"""
def evalModel(model, dataset_train, dataset_val, first_pass):

    if (first_pass == True):
        for batch in dataset_train.take(1):
            inputs, targets = batch

        print("Input shape:", inputs.numpy().shape)
        print("Target shape:", targets.numpy().shape)

        def make_model(input_shape):
            input_layer = keras.layers.Input(input_shape)

            conv1 = keras.layers.Conv1D(filters=16, kernel_size=3, padding="same")(input_layer)
            conv1 = keras.layers.BatchNormalization()(conv1)
            conv1 = keras.layers.ReLU()(conv1)

            conv2 = keras.layers.Conv1D(filters=8, kernel_size=3, padding="same")(conv1)
            conv2 = keras.layers.BatchNormalization()(conv2)
            conv2 = keras.layers.ReLU()(conv2)

            conv3 = keras.layers.Conv1D(filters=4, kernel_size=3, padding="same")(conv2)
            conv3 = keras.layers.BatchNormalization()(conv3)
            conv3 = keras.layers.ReLU()(conv3)

            gap = keras.layers.GlobalAveragePooling1D()(conv3)

            output_layer = keras.layers.Dense(2, activation="softmax")(gap)

            return keras.models.Model(inputs=input_layer, outputs=output_layer)

        model = make_model(inputs.numpy().shape[1:])
        model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["sparse_categorical_accuracy"],
        )
        model.summary()    


    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath="models/OP_classify.keras",
            save_best_only=True, 
            monitor="val_loss"
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=20, min_lr=0.0001
        ),
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=50, verbose=1),
    ]
    
    history = model.fit(
        dataset_train,
        epochs=EPOCHS,    
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        validation_data=dataset_val,
        # verbose=1,
    )

    
    return model, history