import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from mlp_lab import *
from get_Data_Salary import *
from logisticRegression import *

if __name__ == '__main__':
    # Read the data from the file
    data_train = pd.read_csv('SalaryPrediction_train.csv')
    data_test = pd.read_csv('SalaryPrediction_test.csv')
    data_full = pd.read_csv('SalaryPrediction_full.csv')

    data_train_continue = pd.DataFrame()
    data_test_continue = pd.DataFrame()
    data_full_continue = pd.DataFrame()

    data_train_discrete = pd.DataFrame()
    data_test_discrete = pd.DataFrame()
    data_full_discrete = pd.DataFrame()

    for ser in data_train:
        if ser == 'fnl' or ser == 'hpw' or ser == 'gain' or ser == 'edu_int' or ser == 'years' or ser == 'loss' or ser == 'prod':
            data_train_continue[ser] = data_train[ser]
        else:
            data_train_discrete[ser] = data_train[ser]

    for ser in data_test:
        if ser == 'fnl' or ser == 'hpw' or ser == 'gain' or ser == 'edu_int' or ser == 'years' or ser == 'loss' or ser == 'prod':
            data_test_continue[ser] = data_test[ser]
        else:
            data_test_discrete[ser] = data_test[ser]

    for ser in data_full:
        if ser == 'fnl' or ser == 'hpw' or ser == 'gain' or ser == 'edu_int' or ser == 'years' or ser == 'loss' or ser == 'prod':
            data_full_continue[ser] = data_full[ser]
        else:
            data_full_discrete[ser] = data_full[ser]

    # describe_continuous(data_train_continue, data_test_continue, data_full_continue)

    # boxplot_continuous(data_train_continue, data_test_continue, data_full_continue)

    # unique_misses_discrete(data_train_discrete, data_test_discrete, data_full_discrete)

    # histograms_discrete(data_train_discrete, data_test_discrete, data_full_discrete)

    # countplot_data(data_train, data_test)

    # correlations_continuous(data_train_continue, data_test_continue, data_full_continue)

    # correlations_discrete(data_train_discrete, data_test_discrete, data_full_discrete)

    #inputation of missing values
    for ser in data_train_continue:
        imp = SimpleImputer(missing_values=np.nan, strategy='mean')
        data_train_continue[ser] = imp.fit_transform(data_train_continue[ser].values.reshape(-1, 1))
        
    for ser in data_test_continue:
        imp = SimpleImputer(missing_values=np.nan, strategy='mean')
        data_test_continue[ser] = imp.fit_transform(data_test_continue[ser].values.reshape(-1, 1))

    for ser in data_full_continue:
        imp = SimpleImputer(missing_values=np.nan, strategy='mean')
        data_full_continue[ser] = imp.fit_transform(data_full_continue[ser].values.reshape(-1, 1))
   
    for ser in data_train_discrete:
        for i in data_train_discrete[ser]:
            if i == '?':
                data_train_discrete[ser] = data_train_discrete[ser].replace('?', np.nan)
    for ser in data_train_discrete:
        imp = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
        data_train_discrete[ser] = imp.fit_transform(data_train_discrete[ser].values.reshape(-1, 1)).astype(str)
        
    for ser in data_test_discrete:
        for i in data_test_discrete[ser]:
            if i == '?':
                data_test_discrete[ser] = data_test_discrete[ser].replace('?', np.nan)

    for ser in data_test_discrete:
        imp = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
        data_test_discrete[ser] = imp.fit_transform(data_test_discrete[ser].values.reshape(-1, 1)).astype(str)
        

    for ser in data_full_discrete:
        for i in data_full_discrete[ser]:
            if i == '?':
                data_full_discrete[ser] = data_full_discrete[ser].replace('?', np.nan)

    for ser in data_full_discrete:
        imp = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
        data_full_discrete[ser] = imp.fit_transform(data_full_discrete[ser].values.reshape(-1, 1)).astype(str)
  
    # extreme values with iqr and impute with mean
    for ser in data_train_continue:
        Q1 = data_train_continue[ser].quantile(0.25)
        Q3 = data_train_continue[ser].quantile(0.75)
        IQR = Q3 - Q1
        threshold = 1.5
        data_train_continue.loc[((data_train_continue[ser] < (Q1 - threshold * IQR)) | (data_train_continue[ser] > (Q3 + threshold * IQR))), ser] = data_train_continue[ser].mean()

    for ser in data_test_continue:
        Q1 = data_test_continue[ser].quantile(0.25)
        Q3 = data_test_continue[ser].quantile(0.75)
        IQR = Q3 - Q1
        threshold = 1.5
        data_test_continue.loc[((data_test_continue[ser] < (Q1 - threshold * IQR)) | (data_test_continue[ser] > (Q3 + threshold * IQR))), ser] = data_test_continue[ser].mean()
    
    for ser in data_full_continue:
        Q1 = data_full_continue[ser].quantile(0.25)
        Q3 = data_full_continue[ser].quantile(0.75)
        IQR = Q3 - Q1
        threshold = 1.5
        data_full_continue.loc[((data_full_continue[ser] < (Q1 - threshold * IQR)) | (data_full_continue[ser] > (Q3 + threshold * IQR))), ser] = data_full_continue[ser].mean()
   
    # scaling
    scaler = StandardScaler()
    for ser in data_train_continue:
        data_train_continue[ser] = scaler.fit_transform(data_train_continue[ser].values.reshape(-1, 1))

    for ser in data_test_continue:
        data_test_continue[ser] = scaler.fit_transform(data_test_continue[ser].values.reshape(-1, 1))

    for ser in data_full_continue:
        data_full_continue[ser] = scaler.fit_transform(data_full_continue[ser].values.reshape(-1, 1))

    #discrete -label encoding
    for ser in data_train_discrete:
        le = LabelEncoder()
        data_train_discrete[ser] = le.fit_transform(data_train_discrete[ser].values)
    
    for ser in data_test_discrete:
        le = LabelEncoder()
        data_test_discrete[ser] = le.fit_transform(data_test_discrete[ser].values)

    for ser in data_full_discrete:
        le = LabelEncoder()
        data_full_discrete[ser] = le.fit_transform(data_full_discrete[ser].values)

    data_test_discrete = data_test_discrete.astype(float)
    data_train_discrete = data_train_discrete.astype(float)
    data_full_discrete = data_full_discrete.astype(float)
    
    for ser in data_train_continue:
        data_train[ser] = data_train_continue[ser]
        
    for ser in data_train_discrete:
        data_train[ser] = data_train_discrete[ser]

    for ser in data_test_continue:
        data_test[ser] = data_test_continue[ser]
    
    for ser in data_test_discrete:
        data_test[ser] = data_test_discrete[ser]

    for ser in data_full_continue:
        data_full[ser] = data_full_continue[ser]
    
    for ser in data_full_discrete:
        data_full[ser] = data_full_discrete[ser]

    # prepare data
    X = data_full
    
    X = X.drop(columns=['money'])
    X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
    X_train = data_train
    X_train = X_train.drop(columns=['money'])
    X_train = np.concatenate((np.ones((X_train.shape[0], 1)), X_train), axis=1)

    X_test = data_test
    X_test = X_test.drop(columns=['money'])
    X_test = np.concatenate((np.ones((X_test.shape[0], 1)), X_test), axis=1)
    
    y = data_full['money']
    y_train = data_train['money']
    y_test = data_test['money']

    # logistic regression lab implementation
    w, train_nll, test_nll, train_acc, test_acc = train_and_eval_logistic(X, X_train, y_train, X_test, y_test, lr=.01, epochs_no=500)
    Y = predict_logistic(X_test, w)
    
    Y = (Y >= 0.5).astype(int)

    print(f"Acuratete finala pe setul initial - train: {train_acc[-1]}, test: {test_acc[-1]}")

    target_names = ['<= 50K', '> 50K']

    acc = accuracy(y_test, Y)
    print(f"Acuratete pe setul de test: {acc}")
    cnf_matrix = metrics.confusion_matrix(y_test, Y)
    print(cnf_matrix)
    print(classification_report(y_test, Y, target_names=target_names, digits=4))

    #LR sklearn
    model = LogisticRegression(max_iter=700, solver='liblinear')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = metrics.accuracy_score(y_test, y_pred)
    print(f"Acuratete pe setul de testare: {acc}")
    cnf_matrix = metrics.confusion_matrix(y_test, Y)
    print(cnf_matrix)
    print(classification_report(y_test, y_pred, target_names=target_names, digits=4))

    # MLP sklearn
    model = MLPClassifier(max_iter=700, solver='sgd')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = metrics.accuracy_score(y_test, y_pred)
    print(f"Acuratete pe setul de testare MLP: {acc}")
    cnf_matrix = metrics.confusion_matrix(y_test, Y)
    print(cnf_matrix)
    print(classification_report(y_test, y_pred, target_names=target_names, digits=4))

    #apply MLP lab implementation
    BATCH_SIZE = 128
    HIDDEN_UNITS = 300
    EPOCHS_NO = 100

    optimize_args = {'mode': 'SGD', 'lr': .005}

    net = FeedForwardNetwork([Linear(X.shape[1], HIDDEN_UNITS),
                          ReLU(),
                          Linear(HIDDEN_UNITS, 2)])
    cost_function = CrossEntropy()
    y_train = y_train.astype(int)
    y_test = y_test.astype(int)
    for epoch in range(EPOCHS_NO):
        for b_no, idx in enumerate(range(0, len(X_train), BATCH_SIZE)):
            # 1. Pregatim urmatorul batch
            x = X_train[idx:idx + BATCH_SIZE].reshape(-1, X_train.shape[1])
            t = y_train[idx:idx + BATCH_SIZE]
            
            # 2. Calculam gradientul
            # Hint: propagam batch-ul `x` prin reteaua `net`
            #       calculam eroarea pe baza iesirii retelei, folosind `cost_function` 
            #       obtinem gradientul erorii in raport cu iesirea retelei, folosind `backward` pentru `cost_function`
            #       obtinem gradientul in raport cu ponderile retelei `net` folosind `backward` pentru `net`
            y = net.forward(x)
            loss = cost_function.forward(y, t)
            dy = cost_function.backward(y, t)
            net.backward(dy)
            
            # 3. Actualizam parametrii retelei
            net.update(**optimize_args)
        y = net.forward(X_test.reshape(-1, X_test.shape[1]), train=False)
        test_nll = cost_function.forward(y, y_test)
       
    aux = np.argmax(y, axis=1)
    acc = metrics.accuracy_score(y_test, aux)
    print(f"Acuratete pe setul de testare MLP by hand: {acc}")
    cnf_matrix = metrics.confusion_matrix(y_test, aux)
    print(cnf_matrix)
    print(classification_report(y_test, aux, target_names=target_names, digits=4))
