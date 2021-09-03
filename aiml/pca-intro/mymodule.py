"""
@author: drs
"""

def find_missing_value_columns(data):
    missing_value_columns = data.columns[data.isnull().any()]
    print("Columns having missing Values:\n{}".format(missing_value_columns))
    return missing_value_columns;

def find_missing_value_indexes(data):
    missing_value_indexes = data[data.isnull().any(axis=1)].index
    print("Indexes having missing Values:\n{}".format(missing_value_indexes))
    return missing_value_indexes;

def find_outliers_rows(data):
    stats = data.describe()
    # Calculate IQR & the Outlier Whiskers 
    stats.loc['lwhisker'] = (stats.loc['25%']-(stats.loc['75%']-
                                                     stats.loc['25%'])*1.5)
    stats.loc['rwhisker'] = (stats.loc['75%']+(stats.loc['75%']-
                                                     stats.loc['25%'])*1.5)
    print("Data Statistics with Whiskers:\n{}\n\n".format(stats.transpose()))
    # Check for Outliers
    numbervars = stats.columns
    outliers = data[(data[numbervars] > stats[numbervars].loc['rwhisker']) |
                    (data[numbervars] < stats[numbervars].loc['lwhisker'])]
    return outliers;
    

def find_outliers_columns(data):
    outliers = find_outliers_rows(data)
    outliers_colcounts = outliers.count()
    print("Columns with Outlier Counts:\n{}\n\n".format(outliers_colcounts))
    return outliers_colcounts[outliers_colcounts > 0].index;

def find_outliers_indexes(data):
    outliers = find_outliers_rows(data)
    outliers_colcounts = outliers.count()
    outliers_indexes = outliers[outliers_colcounts[
                                                  outliers_colcounts > 0
                                                 ].index
                               ].dropna(how='all').index
    print("Arrt of Indexes that has outluers:\n{}\n\n".format(outliers_indexes))
    print("Summary Statistics of Outlier Data:\n{}\n\n".format(
            data.loc[outliers_indexes].describe()[
                    outliers_colcounts[outliers_colcounts > 0].index].transpose()))
    return outliers_indexes;

def model_and_printscores(model, xtrain, ytrain, xtest, ytest):
    from sklearn import metrics
    import pandas as pd
    print(model)
    model.fit(xtrain, ytrain)
    expected = ytrain
    predicted = model.predict(xtrain)
    train_score = metrics.accuracy_score(expected, predicted)
    train_matrix = metrics.confusion_matrix(expected, predicted)
    expected = ytest
    predicted = model.predict(xtest)
    test_score = metrics.accuracy_score(expected, predicted)
    test_matrix = metrics.confusion_matrix(expected, predicted)
    print("*************************************************************")
    print("Training Accuracy Score: {}\n".format(train_score))
    print("Training Confusion Matrix (Expected, Predicted): \n{}".format(
            pd.DataFrame(train_matrix)))
    print("*************************************************************")
    print("Test Accuracy Score: {}\n".format(test_score))
    print("Test Confusion Matrix (Expected, Predicted): \n{}".format(
            pd.DataFrame(test_matrix)))
    return;
    
def model_and_printcrossvalscores(model, x, y, cval=10):
    from sklearn.model_selection import cross_val_score
    print(model)
    scores = cross_val_score(model, x, y, cv=cval)
    print("*************************************************************")
    print("Cross Validation Scores: \n{}".format(scores))
    print("*************************************************************")
    return;

def get_num_pca_features(data):
    import numpy as np
    import matplotlib.pyplot as plt
    covar = np.cov(data.T)
    eig_vals, eig_vecs = np.linalg.eig(covar)
    eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]
    eig_pairs.sort(reverse=True)
    print("Eigen Pairs in Sorted Order = \n{}".format(eig_pairs))
    variance_explained = (eig_vals / sum(eig_vals))*100
    print("Explained Variance = {}".format(variance_explained))
    plt.plot(np.arange(eig_vals.size)+1, eig_vals, 'ro-', linewidth=2)
    plt.show()
    plt.bar(np.arange(len(variance_explained)), variance_explained,0.5)
    plt.show()
    return((eig_vals > 1).sum());

def pca_3dplot(data):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    fig = plt.figure(1, figsize=(8,6))
    ax = Axes3D(fig, elev=-150, azim=110)
    ax.scatter(data[:,0], data[:,1], data[:,2], c=data.iloc[:,0].values, 
               cmap=plt.cm.Set1, edgecolor='k', s=40)
    ax.set_title("First three PCA directions")
    ax.set_xlabel("1st eigenvector")
    ax.w_xaxis.set_ticklabels([])
    ax.set_ylabel("2nd eigenvector")
    ax.w_yaxis.set_ticklabels([])
    ax.set_zlabel("3rd eigenvector")
    ax.w_zaxis.set_ticklabels([])
    plt.show()
    return;
    
def eliminate_predictors_using_vif(predictors, data, threshold=90):
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    import numpy as np

    for i in np.arange(0,len(predictors)):
        vif = [variance_inflation_factor(data[predictors].values, j) 
        for j in range(data[predictors].shape[1])]
        maxindex = vif.index(max(vif))
        if max(vif) > threshold:
            print ("VIF :", vif)
            print('Eliminating \'' + data[predictors].columns[maxindex] + 
                  '\' at index: ' + str(maxindex))
            del predictors[maxindex]
        else:
            break
    return;