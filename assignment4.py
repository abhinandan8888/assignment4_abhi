import warnings
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.svm import SVC
from itertools import product
from tqdm import tqdm
import matplotlib.pyplot as plt
from skimage.transform import resize
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
warnings.filterwarnings('ignore')
plt.style.use('fivethirtyeight')


def plot_images(X, Y, num_images = 6):
    """
    This function plots images
    """
    plt.figure(figsize=(10,6))
    for i in range(1, num_images+1):
        plt.subplot(int(num_images/3), 3, i)
        plt.imshow(X[i], cmap='gray')
        plt.title("Digit " + str(Y[i]))
        plt.axis('off')
    plt.show()


def data_splitting(X, Y):
    """
    This function splits the data into train, dev, and test set.
    """
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size = 0.92, random_state=42)
    return (X_train,Y_train), (X_test, Y_test)


def hyperparameter_tuning(param_grid, model, model_name, xtrain, xtest, ytrain, cv=5):
    """
    This function performs an extensive hyperparameter tuning
    for a given estimator with parameters list.
    """
    ## grid search cross validation with 5 fold cross validation
    ## and accuracy scoring
    clf = GridSearchCV(model, param_grid, cv=cv, verbose=0, scoring = 'accuracy')
    clf.fit(xtrain, ytrain)
    
    ## get mean and std test score
    mean_test_score = np.mean(clf.cv_results_['mean_test_score'])
    std_test_score = np.mean(clf.cv_results_['std_test_score'])
    
    print(f"model : {model_name}, mean_test_score : {mean_test_score:.4f}, std_test_score : {std_test_score:.4f}")
    
    ## train on best estimator
    final_model = clf.best_estimator_
    final_model.fit(xtrain, ytrain)
    
    ## prediction on test data
    ypred = final_model.predict(xtest)
    
    return ypred


def plots_to_justify_outputs(prediction1, prediction2):
    
    ## get prediction match
    count = 0
    for pred1, pred2 in zip(prediction1, prediction2):
        if pred1 == pred2:
            count += 1

    match_output = count/len(ypred_dt)
    print(f"{match_output * 100 :.2f}% predictions matched in the dataset.")

    ## distribution plot
    sns.distplot(prediction1); sns.distplot(prediction2)
    plt.title('Distribution Plot', fontsize=20)
    plt.show()


if __name__ == '__main__':

    svc_model = SVC()
    svc_params = {'gamma' : ['auto', 'scale'],
                  'C' : [0.1, 0.5, 1, 5, 10, 25, 50, 75, 100, 500, 1000]}

    decision_tree_model = DecisionTreeClassifier(random_state=42)
    decision_tree_params = {'max_depth': [4, 5, 7, 9, 11],
                            'min_samples_split' : [2, 3, 4, 5],
                            'min_samples_leaf' : [1, 3, 5]}




    X_image          = load_digits()['images']
    X_image_reshaped = load_digits()['data']
    Y_image          = load_digits()['target']
    inp_len = len(X_image)
    
    print("Training On Original Images..............................................")
    print(f"Image Size : {X_image[0].shape}")
    plot_images(X_image, Y_image)
    (X_train,Y_train), (X_test, Y_test) = data_splitting(X_image_reshaped, Y_image)

    ypred_dt = hyperparameter_tuning(decision_tree_params, decision_tree_model,
                                     'Decision Tree Classifier', X_train, X_test,
                                     Y_train, cv=5)
    ypred_svc = hyperparameter_tuning(svc_params, svc_model,
                                     'Support Vector Machine', X_train, X_test,
                                     Y_train, cv=5)
    
    plots_to_justify_outputs(ypred_dt, ypred_svc)