import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy
import seaborn as sns
from scipy.stats import chi2_contingency

def describe_continuous(data_continuous_train, data_continuous_test, data_continuous_full):
    with open('data_continue.txt', 'w') as f:
        f.write('Data Train\n')
        f.write(str(data_continuous_train.describe().T))
        f.write('\n\n')
        f.write('Data Test\n')
        f.write(str(data_continuous_test.describe().T))
        f.write('\n\n')
        f.write('Data Full\n')
        f.write(str(data_continuous_full.describe().T))

def boxplot(data_continuous_train, data_continuous_test, data_continuous_full):
    bx = pd.DataFrame.boxplot(data_continuous_train, rot=90, figsize=(20, 20), fontsize=12)
    bx.get_figure().savefig('boxplot_train.png')
    bx = pd.DataFrame.boxplot(data_continuous_test, rot=90, figsize=(20, 20), fontsize=12)
    bx.get_figure().savefig('boxplot_test.png')
    bx = pd.DataFrame.boxplot(data_continuous_full, rot=90, figsize=(20, 20), fontsize=12)
    bx.get_figure().savefig('boxplot_full.png')

def unique_miss_discrete(data_discrete_train, data_discrete_test, data_discrete_full):
    unique_table_train = {}
    not_missing_values_train = {}

    unique_table_test = {}
    not_missing_values_test = {}

    unique_table_full = {}
    not_missing_values_full = {}

    for ser in data_discrete_train:
        unique_table_train[ser] = pd.unique(data_discrete_train[ser])
        nr = data_discrete_train[ser].isnull().sum()
        not_missing_values_train[ser] = len(data_discrete_train[ser]) - nr

    for ser in data_discrete_test:
        unique_table_test[ser] = pd.unique(data_discrete_test[ser])
        nr = data_discrete_test[ser].isnull().sum()
        not_missing_values_test[ser] = len(data_discrete_test[ser]) - nr

    for ser in data_discrete_full:
        unique_table_full[ser] = pd.unique(data_discrete_full[ser])
        nr = data_discrete_full[ser].isnull().sum()
        not_missing_values_full[ser] = len(data_discrete_full[ser]) - nr

    with open('data_discrete.txt', 'w') as f:
        f.write('Data Train\n')
        f.write('Unique values: \n')
        for ser in data_discrete_train:
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
        for ser in data_discrete_test:
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
        for ser in data_discrete_full:
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

def plot_discrete_histograms(data_discrete_full, data_discrete_train, data_discrete_test):
    for ser in data_discrete_train:
        vec = data_discrete_train.value_counts(ser)
        ax = vec.plot(kind='bar', width=0.7, figsize=(10, 12))
        ax.set_xlabel('Valoare')
        ax.set_ylabel('Frecvență')
        ax.set_title('Histogramă cu frecvența apariției valorilor pentru ' + ser)
        plt.savefig('histogram_train_' + ser + '.png')

    for ser in data_discrete_test:
        vec = data_discrete_test.value_counts(ser)
        ax = vec.plot(kind='bar', width=0.7, figsize=(10, 12))
        ax.set_xlabel('Valoare')
        ax.set_ylabel('Frecvență')
        ax.set_title('Histogramă cu frecvența apariției valorilor pentru ' + ser)
        plt.savefig('histogram_test_' + ser + '.png')
    
    for ser in data_discrete_full:
        vec = data_discrete_full.value_counts(ser)
        ax = vec.plot(kind='bar', width=0.7, figsize=(10, 12))
        ax.set_xlabel('Valoare')
        ax.set_ylabel('Frecvență')
        ax.set_title('Histogramă cu frecvența apariției valorilor pentru ' + ser)
        plt.savefig('histogram_full_' + ser + '.png')

def countplot_discrete(data_train, data_test):
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

def correlation_continuous(data_continuous_train, data_continuous_test, data_continuous_full):
    correlation_train = data_continuous_train[['mean_blood_sugar_level', 'body_mass_indicator', 'years_old', 'analysis_results', 'biological_age_index']].corr(method='pearson')


    fig = plt.figure(figsize=(10,10))

    # 111: 1x1 grid, first subplot
    ax = fig.add_subplot(111)

    # normalize data using vmin, vmax
    cax = ax.matshow(correlation_train, vmin=0, vmax=1)

    # add a colorbar to a plot.
    fig.colorbar(cax)

    # define ticks
    ticks = numpy.arange(0,5,1)

    # set x and y tick marks
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)

    # set x and y tick labels
    ax.set_xticklabels(['mean_blood_sugar_level', 'body_mass_indicator', 'years_old', 'analysis_results', 'biological_age_index'])
    ax.set_yticklabels(['mean_blood_sugar_level', 'body_mass_indicator', 'years_old', 'analysis_results', 'biological_age_index'])

    # draw a matrix using the correlations data
    plt.savefig('correlation_train.png')

    correlation_test = data_continuous_test[['mean_blood_sugar_level', 'body_mass_indicator', 'years_old', 'analysis_results', 'biological_age_index']].corr(method='pearson')


    fig = plt.figure(figsize=(10,10))

    # 111: 1x1 grid, first subplot
    ax = fig.add_subplot(111)

    # normalize data using vmin, vmax
    cax = ax.matshow(correlation_train, vmin=0, vmax=1)

    # add a colorbar to a plot.
    fig.colorbar(cax)

    # define ticks
    ticks = numpy.arange(0,5,1)

    # set x and y tick marks
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)

    # set x and y tick labels
    ax.set_xticklabels(['mean_blood_sugar_level', 'body_mass_indicator', 'years_old', 'analysis_results', 'biological_age_index'])
    ax.set_yticklabels(['mean_blood_sugar_level', 'body_mass_indicator', 'years_old', 'analysis_results', 'biological_age_index'])

    # draw a matrix using the correlations data
    plt.savefig('correlation_test.png')

    correlation_train = data_continuous_full[['mean_blood_sugar_level', 'body_mass_indicator', 'years_old', 'analysis_results', 'biological_age_index']].corr(method='pearson')
    # correlation_test = data_test.corr()
    # correlation_full = data_full.corr()

    fig = plt.figure(figsize=(10,10))

    # 111: 1x1 grid, first subplot
    ax = fig.add_subplot(111)

    # normalize data using vmin, vmax
    cax = ax.matshow(correlation_train, vmin=0, vmax=1)

    # add a colorbar to a plot.
    fig.colorbar(cax)

    # define ticks
    ticks = numpy.arange(0,5,1)

    # set x and y tick marks
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)

    # set x and y tick labels
    ax.set_xticklabels(['mean_blood_sugar_level', 'body_mass_indicator', 'years_old', 'analysis_results', 'biological_age_index'])
    ax.set_yticklabels(['mean_blood_sugar_level', 'body_mass_indicator', 'years_old', 'analysis_results', 'biological_age_index'])

    # draw a matrix using the correlations data
    plt.savefig('correlation_full.png')

def correlation_discrete(data_discrete_train, data_discrete_test, data_discrete_full):
    data_train_corr = pd.DataFrame(index=data_discrete_train.columns, columns=data_discrete_train.columns)

    for ser1 in data_discrete_train:
        for ser2 in data_discrete_train:
            CrosstabResult=pd.crosstab(index=data_discrete_train[ser1],columns=data_discrete_train[ser2])
            ChiSqResult = chi2_contingency(CrosstabResult)
            data_train_corr.loc[ser1, ser2] = ChiSqResult[1]

    data_train_corr = data_train_corr.astype(float)

    # print(data_train_corr)
    fig = plt.figure(figsize=(10,10))

    # 111: 1x1 grid, first subplot
    ax = fig.add_subplot(111)

    # normalize data using vmin, vmax
    cax = ax.matshow(data_train_corr, vmin=0, vmax=1)

    # add a colorbar to a plot.
    fig.colorbar(cax)

    # define ticks
    ticks = numpy.arange(0,9,1)

    # set x and y tick marks
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)

    # set x and y tick labels
    # col_discrete = []
    ax.set_xticklabels(['cardiovascular_issues', 'job_category', 'sex', 'tobacco_usage', 'high_blood_pressure', 'married', 'living_area', 'chaotic_sleep', 'cerebrovascular_accident'])
    ax.set_yticklabels(['cardiovascular_issues', 'job_category', 'sex', 'tobacco_usage', 'high_blood_pressure', 'married', 'living_area', 'chaotic_sleep', 'cerebrovascular_accident'])

    # draw a matrix using the correlations data
    plt.savefig('train_discrete_corr.png')

    data_test_corr = pd.DataFrame(index=data_discrete_test.columns, columns=data_discrete_test.columns)

    for ser1 in data_discrete_test:
        for ser2 in data_discrete_test:
            CrosstabResult=pd.crosstab(index=data_discrete_test[ser1],columns=data_discrete_test[ser2])
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
    ticks = numpy.arange(0,9,1)

    # set x and y tick marks
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)

    # set x and y tick labels
    # col_discrete = []
    ax.set_xticklabels(['cardiovascular_issues', 'job_category', 'sex', 'tobacco_usage', 'high_blood_pressure', 'married', 'living_area', 'chaotic_sleep', 'cerebrovascular_accident'])
    ax.set_yticklabels(['cardiovascular_issues', 'job_category', 'sex', 'tobacco_usage', 'high_blood_pressure', 'married', 'living_area', 'chaotic_sleep', 'cerebrovascular_accident'])

    # draw a matrix using the correlations data
    plt.savefig('test_discrete_corr.png')

    data_full_corr = pd.DataFrame(index=data_discrete_full.columns, columns=data_discrete_full.columns)
    for ser1 in data_discrete_full:
        for ser2 in data_discrete_full:
            CrosstabResult=pd.crosstab(index=data_discrete_full[ser1],columns=data_discrete_full[ser2])
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
    ticks = numpy.arange(0,9,1)

    # set x and y tick marks
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)

    # set x and y tick labels
    # col_discrete = []
    ax.set_xticklabels(['cardiovascular_issues', 'job_category', 'sex', 'tobacco_usage', 'high_blood_pressure', 'married', 'living_area', 'chaotic_sleep', 'cerebrovascular_accident'])
    ax.set_yticklabels(['cardiovascular_issues', 'job_category', 'sex', 'tobacco_usage', 'high_blood_pressure', 'married', 'living_area', 'chaotic_sleep', 'cerebrovascular_accident'])

    # draw a matrix using the correlations data
    plt.savefig('full_discrete_corr.png')