import tensorflow as tf
import logging


def get_VGG16_model(input_shape: list, model_path: str) -> tf.keras.models.Model:
    # """
    # Saving and returning the base model extracted from VGG16 model
    # Args:
    #     input_shape (list): shape of the input images
    #     model_path (str): path to save the base model
    # Returns:
    #     tf.python.keras.engine.functional.Functional: Return Type
    # """
    model = tf.keras.applications.vgg16.VGG16(input_shape=input_shape, weights='imagenet', include_top=False)
    model.save(model_path)
    logging.info(f"VGG16 model saved at: {model_path}")
    return model

def prepare_full_model(base_model, learning_rate, CLASSES=2, freeze_all=True, freeze_till=None) -> tf.keras.models.Model:
    # """Prepares the complete transfer learning model
    # Args:
    #     base_model (tf.keras.models.Model): base VGG16 model
    #     learning_rate (float): learning rate
    #     CLASSES (int, optional):  Defaults to 2.
    #     freeze_all (bool, optional): _description_. Defaults to True.
    #     freeze_till (_type_, optional): _description_. Defaults to None.

    # Returns:
    #     tf.keras.models.Model: _description_
    # """
    if freeze_all:
        for layer in base_model.layers:
            layer.trainable = False
    elif (freeze_till is not None) and (freeze_till > 0):
        for layer in base_model.layers[:-freeze_till]:
            layer.trainable = False

    ## Add new layers in the existing base model
    flatten_in = tf.keras.layers.Flatten()(base_model.output)

    prediction = tf.keras.layers.Dense(units = CLASSES, activation='softmax')(flatten_in)

    full_model = tf.keras.models.Model(inputs = base_model.input, outputs = prediction)

    full_model.compile(optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate), loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])

    logging.info(f"Custom model is compiled and ready to train")

    full_model.summary()

    return full_model