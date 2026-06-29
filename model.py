import tensorflow as tf
from tensorflow.keras import layers, Model

def attention_block(x):
    """Channel Attention Mechanism"""
    filters = x.shape[-1]
    avg_pool = layers.GlobalAveragePooling2D()(x)
    max_pool = layers.GlobalMaxPooling2D()(x)
    
    avg_pool = layers.Reshape((1, 1, filters))(avg_pool)
    max_pool = layers.Reshape((1, 1, filters))(max_pool)
    
    dense1_avg = layers.Dense(filters // 8, activation='relu')(avg_pool)
    dense2_avg = layers.Dense(filters, activation='sigmoid')(dense1_avg)
    
    dense1_max = layers.Dense(filters // 8, activation='relu')(max_pool)
    dense2_max = layers.Dense(filters, activation='sigmoid')(dense1_max)
    
    attention = layers.Add()([dense2_avg, dense2_max])
    attention = layers.Activation('sigmoid')(attention)
    
    return layers.Multiply()([x, attention])

def build_model(num_classes=4, input_shape=(224, 224, 3)):
    inputs = layers.Input(shape=input_shape)
    
    # Block 1
    x = layers.Conv2D(32, (3,3), padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(32, (3,3), padding='same', activation='relu')(x)
    x = layers.MaxPooling2D(2,2)(x)
    x = layers.Dropout(0.25)(x)
    
    # Block 2
    x = layers.Conv2D(64, (3,3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, (3,3), padding='same', activation='relu')(x)
    x = layers.MaxPooling2D(2,2)(x)
    x = layers.Dropout(0.25)(x)
    
    # Block 3 + Attention
    x = layers.Conv2D(128, (3,3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = attention_block(x)  # Attention இங்க apply ஆகுது
    x = layers.MaxPooling2D(2,2)(x)
    x = layers.Dropout(0.25)(x)
    
    # Classifier
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs, outputs)
    return model

if __name__ == "__main__":
    model = build_model()
    model.summary()