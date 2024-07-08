import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import chi2_contingency
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from mlp_lab import *

unique_table_train = {}
not_missing_values_train = {}

unique_table_test = {}
not_missing_values_test = {}

unique_table_full = {}
not_missing_values_full = {}

def describe_continuous(data_train_continue, data_test_continue, data_full_continue):
    with open('data_continue.txt', 'w') as f:
        f.write('Data Train\n')
        f.write(str(data_train_continue.describe().T))
        f.write('\n\n')
        f.write('Data Test\n')
        f.write(str(data_test_continue.describe().T))
        f.write('\n\n')
        f.write('Data Full\n')
        f.write(str(data_full_continue.describe().T))

def boxplot_continuous(data_train_continue, data_test_continue, data_full_continue):
    # bx = pd.DataFrame.boxplot(data_train_continue, rot=90, figsize=(10, 10), fontsize=12)
    # bx.get_figure().savefig('boxplot_train.png')


    # bx = pd.DataFrame.boxplot(data_test_continue, rot=90, figsize=(10, 10), fontsize=12)
    # bx.get_figure().savefig('boxplot_test.png')
    bx = pd.DataFrame.boxplot(data_full_continue, rot=90, figsize=(10, 10), fontsize=12)
    bx.get_figure().savefig('boxplot_full.png')

def unique_misses_discrete(data_train_discrete, data_test_discrete, data_full_discrete):
    

    for ser in data_train_discrete:
            unique_table_train[ser] = pd.unique(data_train_discrete[ser])
            nr = data_train_discrete[ser].isnull().sum()
            not_missing_values_train[ser] = len(data_train_discrete[ser]) - nr
            
    for ser in data_test_discrete:
            unique_table_test[ser] = pd.unique(data_test_discrete[ser])
            nr = data_test_discrete[ser].isnull().sum()
            not_missing_values_test[ser] = len(data_test_discrete[ser]) - nr

    for ser in data_full_discrete:
            unique_table_full[ser] = pd.unique(data_full_discrete[ser])
            nr = data_full_discrete[ser].isnull().sum()
            not_missing_values_full[ser] = len(data_full_discrete[ser]) - nr


    with open('data_discrete.txt', 'w') as f:
            f.write('Data Train\n')
            f.write('Unique values: \n')
            for ser in data_train_discrete:
                f.write(ser)
                f.write(': ')
                f.write(str(len(unique_table_train[ser])))
                f.write('\n')
            f.write('\n')
            f.write('Not missing values: \n')
            for ser in not_missing_values_train:
                f.write(ser)
                f.write(': ')
                f.write(str(not_missing_values_train[ser]))
                f.write('\n')
            f.write('\n')

            f.write('Data Test\n')
            f.write('Unique values: \n')
            for ser in data_test_discrete:
                f.write(ser)
                f.write(': ')
                f.write(str(len(unique_table_test[ser])))
                f.write('\n')
            f.write('\n')
            f.write('Not missing values: \n')
            for ser in not_missing_values_test:
                f.write(ser)
                f.write(': ')
                f.write(str(not_missing_values_test[ser]))
                f.write('\n')
            f.write('\n')

            f.write('Data Full\n')
            f.write('Unique values: \n')
            for ser in data_full_discrete:
                f.write(ser)
                f.write(': ')
                f.write(str(len(unique_table_full[ser])))
                f.write('\n')
            f.write('\n')
            f.write('Not missing values: \n')
            for ser in not_missing_values_full:
                f.write(ser)
                f.write(': ')
                f.write(str(not_missing_values_full[ser]))
                f.write('\n')

def histograms_discrete(data_train_discrete, data_test_discrete, data_full_discrete):
    for ser in data_train_discrete:
        vec = data_train_discrete.value_counts(ser)
        ax = vec.plot(kind='bar', width=0.7, figsize=(10, 10))
        ax.set_xlabel('Valoare')
        ax.set_ylabel('Frecvență')
        ax.set_title('Histogramă cu frecvența apariției valorilor pentru ' + ser)
        plt.savefig('histogram_train_' + ser + '.png')

    for ser in data_test_discrete:
        vec = data_test_discrete.value_counts(ser)
        ax = vec.plot(kind='bar', width=0.7, figsize=(10, 10))
        ax.set_xlabel('Valoare')
        ax.set_ylabel('Frecvență')
        ax.set_title('Histogramă cu frecvența apariției valorilor pentru ' + ser)
        plt.savefig('histogram_test_' + ser + '.png')

    for ser in data_full_discrete:
        vec = data_full_discrete.value_counts(ser)
        ax = vec.plot(kind='bar', width=0.7, figsize=(10, 10))
        ax.set_xlabel('Valoare')
        ax.set_ylabel('Frecvență')
        ax.set_title('Histogramă cu frecvența apariției valorilor pentru ' + ser)
        plt.savefig('histogram_full_' + ser + '.png')

def countplot_data(data_train, data_test):
    for ser in data_test:
        plt.figure(figsize=(20, 10))
        plt.title(ser)
        sns.countplot(x= data_test[ser])
        plt.savefig('countplot_test_' + ser + '.png')
        plt.close()

    for ser in data_train:
        plt.figure(figsize=(20, 10))
        plt.title(ser)
        sns.countplot(x= data_train[ser])
        plt.savefig('countplot_train_' + ser + '.png')
        plt.close()

def correlations_continuous(data_train_continue, data_test_continue, data_full_continue):
    correlation_train = data_train_continue[['fnl', 'hpw', 'gain', 'loss', 'edu_int', 'years', 'prod']].corr(method='pearson')
    fig = plt.figure(figsize=(10,10))

    # 111: 1x1 grid, first subplot
    ax = fig.add_subplot(111)

    # normalize data using vmin, vmax
    cax = ax.matshow(correlation_train, vmin=0, vmax=1)

    # add a colorbar to a plot.
    fig.colorbar(cax)

    # define ticks
    ticks = np.arange(0,7,1)

    # set x and y tick marks
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)

    # set x and y tick labels
    ax.set_xticklabels(['fnl', 'hpw', 'gain', 'loss', 'edu_int', 'years', 'prod'])
    ax.set_yticklabels(['fnl', 'hpw', 'gain', 'loss', 'edu_int', 'years', 'prod'])

    # draw a matrix using the correlations data
    plt.savefig('correlation_train.png')

    correlation_test = data_test_continue[['fnl', 'hpw', 'gain', 'loss', 'edu_int', 'years', 'prod']].corr(method='pearson')
    # correlation_test = data_test.corr()
    # correlation_full = data_full.corr()

    fig = plt.figure(figsize=(10,10))

    # 111: 1x1 grid, first subplot
    ax = fig.add_subplot(111)

    # normalize data using vmin, vmax
    cax = ax.matshow(correlation_test, vmin=0, vmax=1)

    # add a colorbar to a plot.
    fig.colorbar(cax)

    # define ticks
    ticks = np.arange(0,7,1)

    # set x and y tick marks
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)

    # set x and y tick labels
    ax.set_xticklabels(['fnl', 'hpw', 'gain', 'loss', 'edu_int', 'years', 'prod'])
    ax.set_yticklabels(['fnl', 'hpw', 'gain', 'loss', 'edu_int', 'years', 'prod'])

    # draw a matrix using the correlations data
    plt.savefig('correlation_test.png')

    correlation_train = data_full_continue[['fnl', 'hpw', 'gain', 'loss', 'edu_int', 'years', 'prod']].corr(method='pearson')
  
    fig = plt.figure(figsize=(10,10))

    # 111: 1x1 grid, first subplot
    ax = fig.add_subplot(111)

    # normalize data using vmin, vmax
    cax = ax.matshow(correlation_train, vmin=0, vmax=1)

    # add a colorbar to a plot.
    fig.colorbar(cax)

    # define ticks
    ticks = np.arange(0,7,1)

    # set x and y tick marks
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)

    # set x and y tick labels
    ax.set_xticklabels(['fnl', 'hpw', 'gain', 'loss', 'edu_int', 'years', 'prod'])
    ax.set_yticklabels(['fnl', 'hpw', 'gain', 'loss', 'edu_int', 'years', 'prod'])

    # draw a matrix using the correlations data
    plt.savefig('correlation_full.png')

def correlations_discrete(data_train_discrete, data_test_discrete, data_full_discrete):
    data_train_corr = pd.DataFrame(index=data_train_discrete.columns, columns=data_train_discrete.columns)

    for ser1 in data_train_discrete:
        for ser2 in data_train_discrete:
            CrosstabResult=pd.crosstab(index=data_train_discrete[ser1],columns=data_train_discrete[ser2])
            ChiSqResult = chi2_contingency(CrosstabResult)
            data_train_corr.loc[ser1, ser2] = ChiSqResult[1]

    data_train_corr = data_train_corr.astype(float)

    fig = plt.figure(figsize=(10,10))

    # 111: 1x1 grid, first subplot
    ax = fig.add_subplot(111)

    # normalize data using vmin, vmax
    cax = ax.matshow(data_train_corr, vmin=0, vmax=1)

    # add a colorbar to a plot.
    fig.colorbar(cax)

    # define ticks
    ticks = np.arange(0,10,1)

    # set x and y tick marks
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)

    # set x and y tick labels
    # col_discrete = []
    ax.set_xticklabels(['relation', 'country', 'job', 'work_type', 'partner', 'edu', 'gender', 'race', 'gtype', 'money'])
    ax.set_yticklabels(['relation', 'country', 'job', 'work_type', 'partner', 'edu', 'gender', 'race', 'gtype', 'money'])

    # draw a matrix using the correlations data
    plt.savefig('train_discrete_corr.png')

    data_test_corr = pd.DataFrame(index=data_test_discrete.columns, columns=data_test_discrete.columns)

    for ser1 in data_test_discrete:
        for ser2 in data_test_discrete:
            CrosstabResult=pd.crosstab(index=data_test_discrete[ser1],columns=data_test_discrete[ser2])
            ChiSqResult = chi2_contingency(CrosstabResult)
            data_test_corr.loc[ser1, ser2] = ChiSqResult[1]

    data_test_corr = data_test_corr.astype(float)

    fig = plt.figure(figsize=(10,10))

    # 111: 1x1 grid, first subplot
    ax = fig.add_subplot(111)

    # normalize data using vmin, vmax
    cax = ax.matshow(data_test_corr, vmin=0, vmax=1)

    # add a colorbar to a plot.
    fig.colorbar(cax)

    # define ticks
    ticks = np.arange(0,10,1)

    # set x and y tick marks
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)

    ax.set_xticklabels(['relation', 'country', 'job', 'work_type', 'partner', 'edu', 'gender', 'race', 'gtype', 'money'])
    ax.set_yticklabels(['relation', 'country', 'job', 'work_type', 'partner', 'edu', 'gender', 'race', 'gtype', 'money'])

    # draw a matrix using the correlations data
    plt.savefig('test_discrete_corr.png')

    data_full_corr = pd.DataFrame(index=data_full_discrete.columns, columns=data_full_discrete.columns)
    for ser1 in data_full_discrete:
        for ser2 in data_full_discrete:
            CrosstabResult=pd.crosstab(index=data_full_discrete[ser1],columns=data_full_discrete[ser2])
            ChiSqResult = chi2_contingency(CrosstabResult)
            data_full_corr.loc[ser1, ser2] = ChiSqResult[1]
    
    data_full_corr = data_full_corr.astype(float)

    fig = plt.figure(figsize=(10,10))

    # 111: 1x1 grid, first subplot
    ax = fig.add_subplot(111)

    # normalize data using vmin, vmax
    cax = ax.matshow(data_test_corr, vmin=0, vmax=1)

    # add a colorbar to a plot.
    fig.colorbar(cax)

    # define ticks
    ticks = np.arange(0,10,1)

    # set x and y tick marks
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)

    ax.set_xticklabels(['relation', 'country', 'job', 'work_type', 'partner', 'edu', 'gender', 'race', 'gtype', 'money'])
    ax.set_yticklabels(['relation', 'country', 'job', 'work_type', 'partner', 'edu', 'gender', 'race', 'gtype', 'money'])

    # draw a matrix using the correlations data
    plt.savefig('full_discrete_corr.png')


def logistic(x):
    return 1 / (1 + np.exp(-x))

def nll(Y, T):
    N = T.shape[0]
    aux1 = T * np.log(Y)
    if np.any(1 - Y > 0):
        aux2 = (1 - T) * np.log(1 - Y)
    return -np.sum(aux1 + aux2) / N

def accuracy(Y, T):
    N = Y.shape[0]
    return np.sum((Y >= 0.5) == T) / N

def predict_logistic(X, w):
    N = X.shape[0]
    return logistic(np.dot(X, w))

def train_and_eval_logistic(X, X_train, T_train, X_test, T_test, lr=.01, epochs_no=100):
    (N, D) = X.shape
    w = np.random.randn(D)
    
    train_acc, test_acc = [], []
    train_nll, test_nll = [], []

    for epoch in range(epochs_no):
        # 1. Obtineti Y_train si Y_test folosind functia predict_logistic
        # 2. Adaugati acuratetea si negative log likelihood-ul pentru setul de antrenare si de testare 
        #    la fiecare pas; utilizati functiile accuracy si nll definite anterior
        # 3. Actualizati ponderile w folosind regula de actualizare a gradientului
        Y_train = predict_logistic(X_train, w)
        Y_test = predict_logistic(X_test, w)
        train_acc.append(accuracy(Y_train, T_train))
        test_acc.append(accuracy(Y_test, T_test))
        
        train_nll.append(nll(Y_train, T_train))
        test_nll.append(nll(Y_test, T_test))

        gradient = np.dot(X_train.T, Y_train - T_train) / N
        w = w - lr * gradient
                

    return w, train_nll, test_nll, train_acc, test_acc

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

    model = LogisticRegression(max_iter=700, solver='liblinear')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = metrics.accuracy_score(y_test, y_pred)
    print(f"Acuratete pe setul de testare: {acc}")
    cnf_matrix = metrics.confusion_matrix(y_test, Y)
    print(cnf_matrix)
    print(classification_report(y_test, y_pred, target_names=target_names, digits=4))


    model = MLPClassifier(max_iter=700, solver='sgd')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = metrics.accuracy_score(y_test, y_pred)
    print(f"Acuratete pe setul de testare MLP: {acc}")
    cnf_matrix = metrics.confusion_matrix(y_test, Y)
    print(cnf_matrix)
    print(classification_report(y_test, y_pred, target_names=target_names, digits=4))

    #apply MLP by hand
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
