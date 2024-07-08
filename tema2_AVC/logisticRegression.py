import numpy
def logistic(x):
    return 1 / (1 + numpy.exp(-x))

def nll(Y, T):
    N = T.shape[0]
    aux1 = T * numpy.log(Y)
    if numpy.any(1 - Y != 0):
        aux2 = (1 - T) * numpy.log(1 - Y)
    return -numpy.sum(aux1 + aux2) / N

def accuracy(Y, T):
    N = Y.shape[0]
    return numpy.sum((Y >= 0.5) == T) / N

def precision(Y, T):
    TP = numpy.sum((Y >= 0.5) & (T == 1))
    FP = numpy.sum((Y >= 0.5) & (T == 0))
    return TP / (TP + FP)

def recall(Y, T):
    TP = numpy.sum((Y >= 0.5) & (T == 1))
    FN = numpy.sum((Y < 0.5) & (T == 1))
    return TP / (TP + FN)

def f1_score(Y, T):
    prec = precision(Y, T)
    rec = recall(Y, T)
    return 2 * prec * rec / (prec + rec)

def predict_logistic(X, w):
    N = X.shape[0]
    # print(X)
    return logistic(numpy.dot(X, w))

def train_and_eval_logistic(X, X_train, T_train, X_test, T_test, lr=.01, epochs_no=100):
    #  Antrenati modelul logistic (ponderile W), executand epochs_no pasi din algoritmul de gradient descent
    (N, D) = X.shape
    # print(data_train)
    # Initializare ponderiimport
    w = numpy.random.randn(D)
    
    train_acc, test_acc = [], []
    train_nll, test_nll = [], []

    for epoch in range(epochs_no):
        # 1. Obtineti Y_train si Y_test folosind functia predict_logistic
        # 2. Adaugati acuratetea si negative log likelihood-ul pentru setul de antrenare si de testare 
        #    la fiecare pas; utilizati functiile accuracy si nll definite anterior
        # 3. Actualizati ponderile w folosind regula de actualizare a gradientului
        Y_train = predict_logistic(X_train, w)
        # print(Y_train)
        Y_test = predict_logistic(X_test, w)
        train_acc.append(accuracy(Y_train, T_train))
        test_acc.append(accuracy(Y_test, T_test))
        
        train_nll.append(nll(Y_train, T_train))
        test_nll.append(nll(Y_test, T_test))
        # print(X_train.T)
        print(Y_train - T_train)
        gradient = numpy.dot(X_train.T, Y_train - T_train) / N
        w = w - lr * gradient
        
    return w, train_nll, test_nll, train_acc, test_acc
