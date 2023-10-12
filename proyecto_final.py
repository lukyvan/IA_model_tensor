import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model  # TensorFlow is required for Keras to work
import cv2  
import numpy as np
import matplotlib.pyplot as plt

#Función para categorizar con una imágen de prueba
def categorizar(model,class_names, imagen):
    #plt.imshow(ruta)
    # resize the image
    TAMANO_IMG=224
    imagen = cv2.resize(imagen, (TAMANO_IMG, TAMANO_IMG))
    imagen = np.asarray(imagen)
    # turn the image into a numpy array
    imagen = np.asarray(imagen)
    # Normalize the image
    normalized_image_array = (imagen.astype(np.float32) / 127.5) - 1
    # Create the array of the right shape to feed into the keras model
    # The 'length' or number of images you can put into the array is
    # determined by the first position in the shape tuple, in this case 1
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array

    # Predicts the model
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    # Print prediction and confidence score
    print("Class:", class_name[2:], end="")
    print("Confidence Score:", confidence_score)

def entrenar_modelo(model,class_names):
    trained_model = hub.KerasLayer(model, input_shape=(224, 224, 3))
    datagen = ImageDataGenerator(
        rescale= 1. / 255,
        rotation_range = 10,
        width_shift_range=0.15,
        height_shift_range = 0.15,
        shear_range = 5,
        zoom_range = [0.7, 1.3],
        validation_split = 0.2
        )
    data_gen_entrenamiento = datagen.flow_from_directory("dataset",
                                                        target_size=(224,224),
                                                        batch_size=32, shuffle=True,
                                                        subset="training")
    data_gen_pruebas = datagen.flow_from_directory("dataset",
                                                        target_size=(224,224),
                                                        batch_size=32, shuffle=True,
                                                        subset="validation")
    trained_model = hub.KerasLayer(model, input_shape=(224, 224, 3))
    trained_model.trainable = False
    trained_model = tf.keras.Sequential([
        trained_model,
        tf.keras.layers.Dense(2, activation="softmax")
    ])
    trained_model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    EPOCAS = 2
    entrenamiento = trained_model.fit(
        data_gen_entrenamiento, epochs=EPOCAS, batch_size=32,
        validation_data=data_gen_pruebas
    )
def categorizar_realTime(model,class_names):
    camera = cv2.VideoCapture(0)
    while True:
        # Grab the webcamera's image.
        ret, image = camera.read()
        # Resize the raw image into (224-height,224-width) pixels
        image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
        # Show the image in a window
        cv2.imshow("Webcam Image", image)
        # Make the image a numpy array and reshape it to the models input shape.
        image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)
        # Normalize the image array
        image = (image / 127.5) - 1
        # Predicts the model
        prediction = model.predict(image)
        index = np.argmax(prediction)
        class_name = class_names[index]
        confidence_score = prediction[0][index]
        # Print prediction and confidence score
        print("Class:", class_name[2:], end="")
        print("Confidence Score:", str(np.round(confidence_score * 100))[:-2], "%")
        # Listen to the keyboard for presses.
        keyboard_input = cv2.waitKey(1)
        # 27 is the ASCII for the esc key on your keyboard.
        if keyboard_input == 27:
            break

    camera.release()
    cv2.destroyAllWindows()

def main():
    # Disable scientific notation for clarity
    np.set_printoptions(suppress=True)
    # Load the model
    modelo = load_model('model.savedmodel', compile=False)
    # Load the labels
    nombres_clase = open("labels.txt", "r").readlines()
    entrenar_modelo(modelo, nombres_clase)
    imagen_test = cv2.imread('test1.jpg')
    categorizar(modelo,nombres_clase,imagen_test)
    categorizar_realTime(modelo,nombres_clase)
if __name__ == "__main__":
    main()