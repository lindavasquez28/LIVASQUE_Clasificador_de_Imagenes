import os
import shutil
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

def organizar_directorios(ruta_train):
    print(f"Python está buscando en: {ruta_train}")

    if not os.path.exists(ruta_train):
        print("Python dice que esta ruta no existe. Revisa que esté bien pegada.")
        return

    archivos = [f for f in os.listdir(ruta_train) if f.endswith('.jpg')]
    print(f"Python logró ver {len(archivos)} fotos .jpg en esta carpeta")
    
    if len(archivos) > 0:
        ruta_dog = os.path.join(ruta_train, "dog")
        ruta_cat = os.path.join(ruta_train, "cat")
        
        os.makedirs(ruta_dog, exist_ok=True)
        os.makedirs(ruta_cat, exist_ok=True)
        
        for archivo in archivos:
            ruta_origen = os.path.join(ruta_train, archivo)
            if "dog" in archivo.lower():
                shutil.move(ruta_origen, os.path.join(ruta_dog, archivo))
            elif "cat" in archivo.lower():
                shutil.move(ruta_origen, os.path.join(ruta_cat, archivo))
                
        print("Las fotos han sido ordenadas y separadas.")

#entrenamiento
def entrenar_modelo(carpeta_datos, carpeta_modelos):
    tamano_foto = (200, 200)

    generador_imagenes = ImageDataGenerator(rescale=1./255, validation_split=0.2)

    print("\nCargando datos de entrenamiento...")
    datos_entrenamiento = generador_imagenes.flow_from_directory(
        carpeta_datos,
        target_size=tamano_foto,
        classes=["dog", "cat"],
        batch_size=8,
        subset='training'
    )

    print("Cargando datos de prueba (validacion)...")
    datos_prueba = generador_imagenes.flow_from_directory(
        carpeta_datos,
        target_size=tamano_foto,
        classes=["dog", "cat"],
        batch_size=32,
        subset='validation'
    )

    # arquitectura
    modelo = Sequential()
    modelo.add(Conv2D(input_shape=(200, 200, 3), filters=32, kernel_size=(3,3), padding="same", activation="relu"))
    modelo.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
    modelo.add(Conv2D(filters=64, kernel_size=(3,3), padding="same", activation="relu"))
    modelo.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
    modelo.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
    modelo.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
    modelo.add(Flatten())
    modelo.add(Dense(units=256, activation="relu"))
    modelo.add(Dense(units=2, activation="softmax"))

    modelo.summary()

    # guardado
    modelo.compile(
        loss="categorical_crossentropy", 
        optimizer=Adam(learning_rate=0.001), 
        metrics=["accuracy"]
    )

    os.makedirs(carpeta_modelos, exist_ok=True)
    ruta_guardado = os.path.join(carpeta_modelos, "modelo_gatos_perros.keras")

    punto_guardado = ModelCheckpoint(filepath=ruta_guardado, monitor="val_accuracy", save_best_only=True, verbose=1)
    parada_temprana = EarlyStopping(monitor="val_accuracy", patience=3, verbose=1)

    print("\nIniciando entrenamiento de la red neuronal...")
    modelo.fit(
        datos_entrenamiento,
        validation_data=datos_prueba,
        epochs=5, 
        steps_per_epoch=20,
        validation_steps=5,
        callbacks=[punto_guardado, parada_temprana]
    )
    print("Entrenamiento finalizado.")

# prediccion
def predecir_imagen(ruta_modelo, ruta_foto):
    print(f"\nAnalizando imagen: {ruta_foto}...")
    modelo_guardado = load_model(ruta_modelo)

    foto_prueba = image.load_img(ruta_foto, target_size=(200, 200))
    foto_arreglo = image.img_to_array(foto_prueba) / 255.0
    foto_lista = np.expand_dims(foto_arreglo, axis=0)

    prediccion = modelo_guardado.predict(foto_lista)

    probabilidad_perro = prediccion[0][0] * 100
    probabilidad_gato = prediccion[0][1] * 100

    if probabilidad_perro > probabilidad_gato:
        print(f"El modelo está {probabilidad_perro:.2f}% seguro de que es un perro")
    else:
        print(f"El modelo está {probabilidad_gato:.2f}% seguro de que es un gato")

    # mostrar imagen
    plt.imshow(foto_prueba)
    plt.axis('off')
    plt.show()

# ejecucion
if __name__ == "__main__":
    # Configuración de rutas
    RUTA_DATOS = "/workspaces/LIVASQUE_Clasificador_de_Imagenes/data/raw/dogs-vs-cats/train"
    RUTA_MODELOS = "/workspaces/LIVASQUE_Clasificador_de_Imagenes/models"
    RUTA_MODELO_GUARDADO = os.path.join(RUTA_MODELOS, "modelo_gatos_perros.keras")
    RUTA_FOTO_PRUEBA = "/workspaces/LIVASQUE_Clasificador_de_Imagenes/data/Gato_(2)_REFON.jpg"

    if os.path.exists(RUTA_MODELO_GUARDADO):
        predecir_imagen(RUTA_MODELO_GUARDADO, RUTA_FOTO_PRUEBA)
    else:
        print(f"No se encontró el modelo en {RUTA_MODELO_GUARDADO}. Entrenelo primero")