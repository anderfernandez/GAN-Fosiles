def generador_de_imagenes(tamaño_imagen = 64):

    from tensorflow.keras.models import Model, Sequential
    from tensorflow.keras.layers import Dense, Conv2DTranspose, LeakyReLU, Reshape, BatchNormalization, Conv2D

    generador = Sequential()

    generador.add(Dense(tamaño_imagen*16*16, input_shape = (100,)))
    #generador.add(BatchNormalization())
    generador.add(LeakyReLU())
    generador.add(Reshape((16,16,tamaño_imagen))) #16x16

    generador.add(Conv2DTranspose(128,kernel_size=3, strides=2, padding = "same"))  #32x32
    generador.add(BatchNormalization(momentum=0.8))
    generador.add(LeakyReLU(alpha=0.2))


    generador.add(Conv2DTranspose(128,kernel_size=3, strides=2, padding = "same")) #64x64
    generador.add(BatchNormalization(momentum=0.8))
    generador.add(LeakyReLU(alpha=0.2))

    generador.add(Conv2D(1,kernel_size=3, padding = "same", activation='tanh'))

    return(generador)

def generar_datos_entrada(n_muestras):
    
    import numpy as np
    
    X = np.random.randn(100 * n_muestras)
    X = X.reshape(n_muestras, 100)
    return X

def crear_datos_fake(modelo_generador, n_muestras):
    
    import numpy as np
    from functions import generar_datos_entrada
    
    input = generar_datos_entrada(n_muestras)
    X = modelo_generador.predict(input)
    y = np.zeros((n_muestras, 1))
    return X,y

def discriminador_de_imagenes(tamaño_imagen = 64):

    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, LeakyReLU, Conv2D, Flatten, Dropout
    from tensorflow.keras.optimizers import Adam
    
    discriminador = Sequential()
    discriminador.add(Conv2D(128, kernel_size=3, padding = "same", input_shape = (tamaño_imagen,tamaño_imagen,1)))  #64x64
    discriminador.add(LeakyReLU(alpha=0.2))
    discriminador.add(Dropout(0.2))

    discriminador.add(Conv2D(128, kernel_size=3,strides=(2,2), padding = "same"))  #32x32
    discriminador.add(LeakyReLU(alpha=0.2))
    discriminador.add(Dropout(0.2))

    discriminador.add(Conv2D(128, kernel_size=3,strides=(2,2), padding = "same"))  #16x16
    discriminador.add(LeakyReLU(alpha=0.2))
    discriminador.add(Dropout(0.2))

    discriminador.add(Conv2D(64, kernel_size=3,strides=(2,2), padding = "same")) #8x8
    discriminador.add(LeakyReLU(alpha=0.2))
    discriminador.add(Dropout(0.2))

    discriminador.add(Flatten())
    discriminador.add(Dense(1, activation='sigmoid'))

    opt = Adam(lr=0.0005 ,beta_1=0.5)
    discriminador.compile(loss='binary_crossentropy', optimizer= opt , metrics = ['accuracy'])

    return(discriminador)

def cargar_datos_reales(dataset, n_muestras):

    import numpy as np

    ix = np.random.randint(0, dataset.shape[0], n_muestras)
    X = dataset[ix]
    y = np.ones((n_muestras, 1))
    return X,y

def cargar_datos_fake(n_muestras, tamaño_imagen = 64):

    import numpy as np
    
    X = np.random.rand(tamaño_imagen * tamaño_imagen * 1 * n_muestras)
    X = -1 + X * 2
    X = X.reshape((n_muestras, tamaño_imagen,tamaño_imagen,1))
    y = np.zeros((n_muestras, 1))
    return X,y

def entrenar_discriminador(modelo, dataset, n_iteraciones=10, batch = 128):
    from functions import cargar_datos_reales, cargar_datos_fake
    medio_batch = int(batch/2)

    for i in range(n_iteraciones):
        X_real, y_real = cargar_datos_reales(dataset, medio_batch)
        _, acc_real = modelo.train_on_batch(X_real, y_real)

        X_fake, y_fake = cargar_datos_fake(medio_batch)
        _, acc_fake = modelo.train_on_batch(X_fake, y_fake)

        print(str(i+1) + ' Real:' + str(acc_real*100) + ', Fake:' + str(acc_fake*100))


def crear_gan(discriminador, generador,n_batch = None):
    
    from tensorflow.keras import Input
    from tensorflow.keras.models import Model
    from tensorflow.keras.optimizers import Adam

    discriminador.trainable=False

    gan_input = Input(shape=(n_batch,))
    generator_output = generador(gan_input)
    gan_output = discriminador(generator_output)

    gan = Model(inputs=gan_input, outputs=gan_output)

    opt = Adam(lr=0.0005,beta_1=0.5) 
    gan.compile(loss = "binary_crossentropy", optimizer = opt)

    discriminador.trainable = True
    return gan

def mostrar_imagenes_generadas(datos_fake, epoch):

    from datetime import datetime
    import matplotlib.pyplot as plt

    now = datetime.now()
    now = now.strftime("%Y%m%d_%H%M%S")

    # Hacemos que los datos vayan de 0 a 1
    datos_fake = (datos_fake + 1) / 2.0
    
    for i in range(10):
        plt.imshow(datos_fake[i,:,:,0],cmap = 'gray' )
        plt.axis('off')
        nombre = 'imagenes_generadas/' + str(epoch) + '_imagen_generada_' + str(i) + '.png'
        plt.savefig(nombre, bbox_inches='tight')
        plt.close()

def evaluar_y_guardar(modelo_generador, modelo_discriminador, modelo_gan, epoch, medio_dataset, dataset):
    
    from datetime import datetime
    from tensorflow.keras.models import save_model
    # Guardamos el modelo
    # Creamos los nombres
    now = datetime.now()
    now = now.strftime("%Y%m%d_%H%M%S")
    nombre_gan = 'models/' + str(epoch) + '_' + str(now)+"_gan"
    nombre_generador = 'models/' + str(epoch) + '_' + str(now)+"_generador"
    nombre_discriminador = 'models/' + str(epoch) + '_' + str(now)+"_discriminador"

    # Guardamos los modelos
    modelo_discriminador.trainable = False
    save_model(modelo_gan, nombre_gan ,save_format='tf')
    modelo_discriminador.trainable = True
    save_model(modelo_generador, nombre_generador, save_format='tf')
    save_model(modelo_discriminador, nombre_discriminador, save_format='tf')
    # Generamos nuevos datos
    X_real,Y_real = cargar_datos_reales(dataset, medio_dataset)
    X_fake, Y_fake =  crear_datos_fake(modelo_generador,medio_dataset)

    # Evaluamos el modelo
    _, acc_real = modelo_discriminador.evaluate(X_real, Y_real)
    _, acc_fake = modelo_discriminador.evaluate(X_fake, Y_fake)

    print('Acc Real:' + str(acc_real*100) + '% Acc Fake:' + str(acc_fake*100)+'%')

def entrenamiento(gan, datos, modelo_generador, modelo_discriminador, epochs, n_batch, inicio = 0):
  
  from functions import crear_datos_fake, generar_datos_entrada, cargar_datos_reales
  import numpy as np

  dimension_batch = int(datos.shape[0]/n_batch)
  medio_dataset = int(n_batch/2)

  # Iteramos para todos los epochs
  for epoch in range(inicio, inicio + epochs):
    # Iteramos para todos los batches
    for batch in range(n_batch):

      # Cargamos datos reales
      X_real,Y_real = cargar_datos_reales(datos, medio_dataset)

      # Enrenamos discriminador con datos reales
      coste_discriminador_real, _ = modelo_discriminador.train_on_batch(X_real, Y_real)
      X_fake, Y_fake =  crear_datos_fake(modelo_generador,medio_dataset)

      coste_discriminador_fake, _ = modelo_discriminador.train_on_batch(X_fake, Y_fake)

      # Generamos datos de entadas de la GAN
      X_gan = generar_datos_entrada(medio_dataset)
      Y_gan = np.ones((medio_dataset, 1))

      # Entrenamos la GAN con datos falsos
      coste_gan = gan.train_on_batch(X_gan, Y_gan)

    # Cada 10 Epochs mostramos resultados y el coste
    if (epoch+1) % 1 == 0:
      evaluar_y_guardar(modelo_generador, modelo_discriminador,  gan, epoch, medio_dataset, datos)
      mostrar_imagenes_generadas(X_fake, epoch = epoch)