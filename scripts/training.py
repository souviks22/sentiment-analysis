import numpy as np
import pandas as pd
from tensorflow.keras.layers import Input, Embedding, SpatialDropout1D, Conv1D, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau

def get_model(X_train: pd.Series, y_train: pd.Series, X_test: pd.Series, y_test: pd.Series, embedding: np.array) -> Model:
    input_layer = Input(shape=(35,),dtype='int32')
    embedding_layer = Embedding(
        input_dim=len(embedding),
        output_dim=100,
        weights=[embedding],
        trainable=False
    )(input_layer)

    dropout_layer = SpatialDropout1D(rate=0.2)(embedding_layer)
    convolutional_layer = Conv1D(
        filters=64,
        kernel_size=5,
        activation='relu'
    )(dropout_layer)

    recurrent_layer = Bidirectional(LSTM(
        units=64,
        dropout=0.2,
        recurrent_dropout=0.2
    ))(convolutional_layer)

    dense_layer = Dense(units=512,activation='relu')(recurrent_layer)
    dropout_layer = Dropout(rate=0.5)(dense_layer)
    dense_layer = Dense(units=512,activation='relu')(dropout_layer)
    output_layer = Dense(units=1,activation='sigmoid')(dense_layer)

    model = Model(input_layer,output_layer)
    model.compile(
        optimizer=Adam(learning_rate=10**-2),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    reduce = ReduceLROnPlateau(
        factor=0.1,
        min_lr=10**-5,
        monitor='val_loss',
        verbose=2
    )

    model.fit(
        x=X_train,
        y=y_train,
        batch_size=512,
        epochs=10,
        validation_data=(X_test,y_test),
        callbacks=[reduce]
    )

    model.save('models/lstm_model.keras')
    return model