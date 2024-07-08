import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import chi2_contingency
import pandas as pd

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
    bx = pd.DataFrame.boxplot(data_train_continue, rot=90, figsize=(10, 10), fontsize=12)
    bx.get_figure().savefig('boxplot_train.png')


    bx = pd.DataFrame.boxplot(data_test_continue, rot=90, figsize=(10, 10), fontsize=12)
    bx.get_figure().savefig('boxplot_test.png')
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