from keras.models import Sequential
from keras.layers import Dense
import numpy
from tensorflow.contrib.learn.python.learn.estimators._sklearn import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score


def principal():
    # deteminamos la semilla 7
    numpy.random.seed(7)

    # cargamos el dataset
    dataset = numpy.loadtxt("dataset.csv", delimiter=",")

    # compilamos el modelo
    model=compilarModelo()

    # dividimos el dataset
    X_train, X_test, Y_train, Y_test = dividirDataset(dataset)

    # entrenamos el modelo 150,10
    model.fit(X_train, Y_train, epochs=100, batch_size=10)

    # validamos el modelo
    predicciones = (model.predict(X_test)).round()

    evaluacion = model.evaluate(X_test,Y_test,verbose=0)

    print("evaluacion %s"%evaluacion)

    # Confusion matrix
    print("matriz de confusion")
    matrizC=confusion_matrix(Y_test, predicciones)
    print(matrizC)

    # Accuracy: relacion entre el numero correcto de predicciones y el total de predicciones
    print("accuracy")
    accuracy = accuracy_score(Y_test,predicciones)
    print(accuracy)

    # Precision: indica la precision del clasificador, es decir, de los TP obtenidos cuantos realmente son verdaderos
    print("precision")
    precision=precision_score(Y_test, predicciones)
    print(precision)

    # Recall: indica la cantidad de elementos verdaderos encontrados por el clasificador
    print("recall")
    recall=recall_score(Y_test, predicciones)
    print(recall)

    return model


def dividirDataset(dataset):
    X=dataset[:, 0:21]
    Y= dataset[:, 21]

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.35, random_state = 0)
    print(X_train.shape)
    print(X_test.shape)
    print(Y_train.shape)
    print(Y_test.shape)

    return (X_train, X_test, Y_train, Y_test)

def compilarModelo():
    model = Sequential()
    model.add(Dense(21, input_dim=21, init='uniform', activation='relu'))
    model.add(Dense(15, init='uniform', activation='relu'))
    model.add(Dense(1, init='uniform', activation='sigmoid'))

    model.summary();

    # Compilamos el modelo
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

def evaluarNuevaMedicion(mes,dia,cultivo,eto,kc,tsuelo,mecanismo,eficienciaM,ppe,tempA,humA,presionA,radiacionS,tempS,humS,model):
    cantidadAgua=0
    etc=eto*kc
    p=0.3
    wc=22
    pm=10
    nnr=etc-ppe
    nbr=nnr/eficienciaM
    consulta=numpy.array([mes,dia,cultivo,etc,eto,kc,p,wc,pm,tsuelo,mecanismo,eficienciaM,ppe,nnr,nbr,tempA,humA,presionA,radiacionS,tempS,humS])

    resultado=model.predict(numpy.array([consulta, ])).round()
    print((numpy.array([consulta, ])).shape)
    print(model.predict(numpy.array([consulta, ])))
    print(resultado)
    if(resultado==0):
        print("no hay que regar")
        cantidadAgua=0

    else:
        print("hay que regar")
        print("cantidad de agua %s"%nbr)
        cantidadAgua=nbr

    # hay que guardar la consulta en la base de datos
    consultaNueva=numpy.insert(consulta,21,resultado)

    print((numpy.array([consultaNueva, ])).shape)
    llenarNuevosDatos(consultaNueva)

    return cantidadAgua

def llenarDatasetNuevosDatos():

    dataset = numpy.loadtxt("valoresNuevos.csv", delimiter=",")
    print("hola")
    tamanioDataset = (dataset.size)/22
    i=0

    while(i<tamanioDataset):
        if(tamanioDataset==1):
            datasetActual = dataset
            tamanio = datasetActual.size
        else:
            datasetActual = dataset[i]
            tamanio = datasetActual.size
        consultaGuardar=""
        j=0

        while(j<tamanio):
            if (j == ((datasetActual.__len__() - 1))):
                consultaGuardar = consultaGuardar + str(datasetActual[j])
            else:
                consultaGuardar = consultaGuardar + str(datasetActual[j]) + ","
            j += 1
        llenarDataset(consultaGuardar)
        print(consultaGuardar)
        i+=1

    datosNuevos = open('valoresNuevos.csv','w')
    datosNuevos.truncate()

def llenarDataset(datos):

    dataset=open('dataset.csv','a')
    dataset.writelines("\r")
    dataset.writelines(datos)
    dataset.close()

def llenarNuevosDatos(consultaNueva):
    consultaGuardar = ""
    i = 0
    while (i < consultaNueva.__len__()):
        if (i == ((consultaNueva.__len__() - 1))):
            consultaGuardar = consultaGuardar + str(consultaNueva[i])
        else:
            consultaGuardar = consultaGuardar + str(consultaNueva[i]) + ","
        i += 1
    dataset=open('valoresNuevos.csv','a+')
    dataset.writelines(consultaGuardar)
    dataset.writelines("\r")
    dataset.close()


#llamadas al sistema
modeloProduccion=principal()
resultado=evaluarNuevaMedicion(12,29,1,6.03,1.05,1,1,0.5,0,28,27,1007,7086,51,58,modeloProduccion)
print(resultado)
llenarDatasetNuevosDatos()
