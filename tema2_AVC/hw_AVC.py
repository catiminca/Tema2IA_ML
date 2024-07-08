from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.neural_network import MLPClassifier
from mlp_lab import *
from sklearn.metrics import classification_report
from get_Data_AVC import *
from logisticRegression import *

if __name__ == '__main__':
    data_train = pd.read_csv('AVC_train.csv')
    data_test = pd.read_csv('AVC_test.csv')
    data_full = pd.read_csv('AVC_full.csv')

    data_continuous_train = pd.DataFrame()
    data_discrete_train = pd.DataFrame()

    data_continuous_test = pd.DataFrame()
    data_discrete_test = pd.DataFrame()

    data_continuous_full = pd.DataFrame()
    data_discrete_full = pd.DataFrame()

    for ser in data_train:
        if ser == 'mean_blood_sugar_level' or ser == 'body_mass_indicator' or ser == 'years_old' or ser == 'analysis_results' or ser == 'biological_age_index':

            data_continuous_train[ser] = data_train[ser]
        else:
            data_discrete_train[ser] = data_train[ser]

    for ser in data_test:
        if ser == 'mean_blood_sugar_level' or ser == 'body_mass_indicator' or ser == 'years_old' or ser == 'analysis_results' or ser == 'biological_age_index':
            data_continuous_test[ser] = data_test[ser]
        else:
            data_discrete_test[ser] = data_test[ser]

    for ser in data_full:
        if ser == 'mean_blood_sugar_level' or ser == 'body_mass_indicator' or ser == 'years_old' or ser == 'analysis_results' or ser == 'biological_age_index':
            data_continuous_full[ser] = data_full[ser]
        else:
            data_discrete_full[ser] = data_full[ser]
    

    # describe_continuous(data_continuous_train, data_continuous_test, data_continuous_full)

    # boxplot(data_continuous_train, data_continuous_test, data_continuous_full)
    
    # unique_miss_discrete(data_discrete_train, data_discrete_test, data_discrete_full)

    # plot_discrete_histograms(data_discrete_full, data_discrete_train, data_discrete_test)

    # countplot_discrete(data_train, data_test)

    # correlation_continuous(data_continuous_train, data_continuous_test, data_continuous_full)

    # correlation_discrete(data_discrete_train, data_discrete_test, data_discrete_full)

    # impute nan values
    for ser in data_continuous_train:
        imputer = SimpleImputer(missing_values=numpy.nan, strategy='mean')
        data_continuous_train[ser] = imputer.fit_transform(data_continuous_train[ser].values.reshape(-1, 1))

    for ser in data_continuous_test:
        imputer = SimpleImputer(missing_values=numpy.nan, strategy='mean')
        data_continuous_test[ser] = imputer.fit_transform(data_continuous_test[ser].values.reshape(-1, 1))

    for ser in data_continuous_full:
        imputer = SimpleImputer(missing_values=numpy.nan, strategy='mean')
        data_continuous_full[ser] = imputer.fit_transform(data_continuous_full[ser].values.reshape(-1, 1))

    for ser in data_discrete_train:
        imp = SimpleImputer(missing_values=numpy.nan, strategy='most_frequent')
        data_discrete_train[ser] = imp.fit_transform(data_discrete_train[ser].values.reshape(-1, 1)).astype(str)

    for ser in data_discrete_test:
        imp = SimpleImputer(missing_values=numpy.nan, strategy='most_frequent')
        data_discrete_test[ser] = imp.fit_transform(data_discrete_test[ser].values.reshape(-1, 1)).astype(str)

    for ser in data_discrete_full:
        imp = SimpleImputer(missing_values=numpy.nan, strategy='most_frequent')
        data_discrete_full[ser] = imp.fit_transform(data_discrete_full[ser].values.reshape(-1, 1)).astype(str)

    # remove extreme values
    for ser in data_continuous_train:
        Q1 = data_continuous_train[ser].quantile(0.25)
        Q3 = data_continuous_train[ser].quantile(0.75)
        IQR = Q3 - Q1
        threshold = 1.5
        data_continuous_train.loc[((data_continuous_train[ser] < (Q1 - threshold * IQR)) | (data_continuous_train[ser] > (Q3 + threshold * IQR))), ser] = data_continuous_train[ser].mean()

    for ser in data_continuous_test:
        Q1 = data_continuous_test[ser].quantile(0.25)
        Q3 = data_continuous_test[ser].quantile(0.75)
        IQR = Q3 - Q1
        threshold = 1.5
        data_continuous_test.loc[((data_continuous_test[ser] < (Q1 - threshold * IQR)) | (data_continuous_test[ser] > (Q3 + threshold * IQR))), ser] = data_continuous_test[ser].mean()

    for ser in data_continuous_full:
        Q1 = data_continuous_full[ser].quantile(0.25)
        Q3 = data_continuous_full[ser].quantile(0.75)
        IQR = Q3 - Q1
        threshold = 1.5
        data_continuous_full.loc[((data_continuous_full[ser] < (Q1 - threshold * IQR)) | (data_continuous_full[ser] > (Q3 + threshold * IQR))), ser] = data_continuous_full[ser].mean()
   
    # standardize data
    scaler = StandardScaler()
    for ser in data_continuous_train:
        data_continuous_train[ser] = scaler.fit_transform(data_continuous_train[ser].values.reshape(-1, 1))

    for ser in data_continuous_test:
        data_continuous_test[ser] = scaler.fit_transform(data_continuous_test[ser].values.reshape(-1, 1))

    for ser in data_continuous_full:
        data_continuous_full[ser] = scaler.fit_transform(data_continuous_full[ser].values.reshape(-1, 1))
    
    for ser in data_discrete_train:
        le = LabelEncoder()
        data_discrete_train[ser] = le.fit_transform(data_discrete_train[ser].values)

    for ser in data_discrete_test:
        le = LabelEncoder()
        data_discrete_test[ser] = le.fit_transform(data_discrete_test[ser].values)

    for ser in data_discrete_full:
        le = LabelEncoder()
        data_discrete_full[ser] = le.fit_transform(data_discrete_full[ser].values)

    #copy data
    for ser in data_continuous_train:
        data_train[ser] = data_continuous_train[ser]

    for ser in data_continuous_test:
        data_test[ser] = data_continuous_test[ser]

    for ser in data_continuous_full:
        data_full[ser] = data_continuous_full[ser]

    for ser in data_discrete_train:
        data_train[ser] = data_discrete_train[ser]

    for ser in data_discrete_test:
        data_test[ser] = data_discrete_test[ser]

    for ser in data_discrete_full:
        data_full[ser] = data_discrete_full[ser]

    # prepare data for training
    X = data_full
    X_train = data_train
    X_test = data_test
    X = X.drop(columns=['cerebrovascular_accident'])
    X = numpy.concatenate([X, numpy.ones((X.shape[0], 1))], axis=1)

    X_test = X_test.drop(columns=['cerebrovascular_accident'])
    X_test = numpy.concatenate([X_test, numpy.ones((X_test.shape[0], 1))], axis=1)

    X_train = X_train.drop(columns=['cerebrovascular_accident'])
    X_train = numpy.concatenate([X_train, numpy.ones((X_train.shape[0], 1))], axis=1)

    y = data_full['cerebrovascular_accident']
    y_train = data_train['cerebrovascular_accident']
    y_test = data_test['cerebrovascular_accident']

    #lab implementation
    w, train_nll, test_nll, train_acc, test_acc = train_and_eval_logistic(X, X_train, y_train, X_test, y_test, lr=.01, epochs_no=500)
    Y = predict_logistic(X_test, w)   
    
    Y = (Y >= 0.5).astype(int)
    print(f"Acuratete finala pe setul initial - train: {train_acc[-1]}, test: {test_acc[-1]}")

    target_names = ['without avc', 'with avc']

    acc = accuracy(y_test, Y)
    print(f"Acuratete pe setul de test: {acc}")
    cnf_matrix = metrics.confusion_matrix(y_test, Y)
    print(cnf_matrix)
    print(classification_report(y_test, Y, target_names=target_names, digits=4))

    #sklearn implementation
    model = LogisticRegression(max_iter=700, solver='liblinear')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = metrics.accuracy_score(y_test, y_pred)
    print(f"Acuratete pe setul de test: {acc}")
    cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
    print(cnf_matrix)
    print(classification_report(y_test, y_pred, target_names=target_names, digits=4))

    #MLP
    model = MLPClassifier(max_iter=700, solver='sgd')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = metrics.accuracy_score(y_test, y_pred)
    print(f"Acuratete pe setul de testare MLP: {acc}")
    cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
    print(cnf_matrix)
    print(classification_report(y_test, y_pred, target_names=target_names, digits=4, zero_division=0))

    #apply MLP lab implementation
    BATCH_SIZE = 128
    HIDDEN_UNITS = 300
    EPOCHS_NO = 20

    optimize_args = {'mode': 'SGD', 'lr': .005}

    net = FeedForwardNetwork([Linear(X_train.shape[1], HIDDEN_UNITS),
                          ReLU(),
                          Linear(HIDDEN_UNITS, 2)])
    cost_function = CrossEntropy()
   
    for epoch in range(EPOCHS_NO):
        for b_no, idx in enumerate(range(0, len(X_train), BATCH_SIZE)):
            # 1. Pregatim urmatorul batch
            x = X_train[idx:idx + BATCH_SIZE].reshape(-1, 14)
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

    aux = numpy.argmax(y, axis=1)
    acc = metrics.accuracy_score(y_test, aux)
    print(f"Acuratete pe setul de testare MLP by hand: {acc}")
    cnf_matrix = metrics.confusion_matrix(y_test, aux)
    print(cnf_matrix)
    print(classification_report(y_test, aux, target_names=target_names, digits=4, zero_division=0))

    