from turtle import shape
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, accuracy_score
import xgboost as xgb
from hyperopt import STATUS_OK, hp, fmin, tpe
from feature_engineering import data_process
import os
from time import time
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
import pandas as pd
 
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LinearRegression


## maybe we don't need to change this part
def BO_TPE(X_train, y_train, X_val, y_val):
    "Hyperparameter optimization"
    # xgb.DMatrix(): load a NumPy array into DMatrix

    # model = XGBClassifier()
    scaler = StandardScaler()
    x_trainScaled = scaler.fit_transform(X_train)
    x_valScaled = scaler.fit_transform(X_val)
    # model.fit(x_trainScaled, y_train)


    # train = xgb.DMatrix(X_train, label=y_train)
    # val = xgb.DMatrix(X_val, label=y_val)

    train = xgb.DMatrix(x_trainScaled, label=y_train)
    val = xgb.DMatrix(x_valScaled, label=y_val)

    X_val_D = xgb.DMatrix(X_val)

    def objective(params):
        xgb_model = xgb.train(params, dtrain=train, num_boost_round=1000, evals=[(val, 'eval')],
                              verbose_eval=False, early_stopping_rounds=80)
        y_vd_pred = xgb_model.predict(X_val_D, ntree_limit=xgb_model.best_ntree_limit)
        y_val_class = [0 if i <= 0.5 else 1 for i in y_vd_pred]

        acc = accuracy_score(y_val, y_val_class)
        loss = 1 - acc

        return {'loss': loss, 'params': params, 'status': STATUS_OK}

    max_depths = [3, 4]
    learning_rates = [0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.15, 0.2]
    subsamples = [0.5, 0.6, 0.7, 0.8, 0.9]
    colsample_bytrees = [0.5, 0.6, 0.7, 0.8, 0.9]
    reg_alphas = [0.0, 0.005, 0.01, 0.05, 0.1]
    reg_lambdas = [0.8, 1, 1.5, 2, 4]

    space = {
        'max_depth': hp.choice('max_depth', max_depths),
        'learning_rate': hp.choice('learning_rate', learning_rates),
        'subsample': hp.choice('subsample', subsamples),
        'colsample_bytree': hp.choice('colsample_bytree', colsample_bytrees),
        'reg_alpha': hp.choice('reg_alpha', reg_alphas),
        'reg_lambda': hp.choice('reg_lambda', reg_lambdas),
    }

    best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=20)

    best_param = {'max_depth': max_depths[(best['max_depth'])],
                  'learning_rate': learning_rates[(best['learning_rate'])],
                  'subsample': subsamples[(best['subsample'])],
                  'colsample_bytree': colsample_bytrees[(best['colsample_bytree'])],
                  'reg_alpha': reg_alphas[(best['reg_alpha'])],
                  'reg_lambda': reg_lambdas[(best['reg_lambda'])]
                  }

    print('best_param:\n',best_param)

    return best_param

def train_model(k, X_train, y_train, X_val, y_val, save_model_dir):
    print('*************************************************************')
    print('{}th training ..............'.format(k + 1))
    print('Hyperparameters optimization using  BO_TPE()......')
    start_train = time()
    
 

    print(X_train.shape)
    best_param = BO_TPE(X_train, y_train, X_val, y_val)
    xgb_model = xgb.XGBClassifier(max_depth = best_param['max_depth'],
                                  eta = best_param['learning_rate'],
                                  n_estimators = 1000,
                                  subsample = best_param['subsample'],
                                  colsample_bytree = best_param['colsample_bytree'],
                                  reg_alpha = best_param['reg_alpha'],
                                  reg_lambda = best_param['reg_lambda'],
                                  objective = "binary:logistic"
                                  )
    print('Hyperparameters optimization done!')
    print('Fitting the xgb model ......')
    xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric='error',
                  early_stopping_rounds=80, verbose=False)
    print('Fitting of xgb done!')

    print('Predicting probabilities (training)......')

    y_tr_pred = (xgb_model.predict_proba(X_train, ntree_limit=xgb_model.best_ntree_limit))[:, 1]
    print('Calculating ROC AUC .......')
    train_auc = roc_auc_score(y_train, y_tr_pred)


    print('training dataset AUC: ' + str(train_auc))
    y_tr_class = [0 if i <= 0.5 else 1 for i in y_tr_pred]
    acc = accuracy_score(y_train, y_tr_class)
    print('training dataset acc: ' + str(acc))

    print('Time taken to train {}th training:'.format(k + 1),time() - start_train, 'seconds')


    y_vd_pred = (xgb_model.predict_proba(X_val, ntree_limit=xgb_model.best_ntree_limit))[:, 1]
    valid_auc = roc_auc_score(y_val, y_vd_pred)
    print('validation dataset AUC: ' + str(valid_auc))
    y_val_class = [0 if i <= 0.5 else 1 for i in y_vd_pred]
    acc = accuracy_score(y_val, y_val_class)
    print('validation dataset acc: ' + str(acc))



    print('************************************************************')
    # save the model
    save_model_path = save_model_dir + 'model{}.mdl'.format(k + 1)
    xgb_model.get_booster().save_model(fname=save_model_path)




def downsample(data_set, data_dir,cols_to_remove):
    """
    Using our feature extraction approach will result in over 1 million hours of data in the training process.
    However, only roughly 1.8% of these data corresponds to a positive outcome.
    Consequently, in order to deal with the serious class imbalance, a systematic way is provided by
    down sampling the excessive data instances of the majority class in each cross validation.
    """

    ## data_set: .npy files; data_dir, the directory of the .csv files
    start_data_process = time()
    #code here.
    

    x, y = data_process(data_set, data_dir,cols_to_remove)
    # print(x)
    # print(y)

    print('Time taken to run data_process():',time() - start_data_process, 'seconds')
    index_0 = np.where(y == 0)[0]
    index_1 = np.where(y == 1)[0]
    # print('index_0:', index_0)
    # print('index_1:', index_1)
    ## the rows of septic data must be smaller than that of nonseptic
    ## so len(index_1) must be within the range of len(index_0)
    index = index_0[len(index_1): -1]
    # print('index:',index)
    x_del = np.delete(x, index, 0)
    y_del = np.delete(y, index, 0)

    # print('x_del:',x_del)
    # print('y_del:',y_del)

    index = [i for i in range(len(y_del))]
    np.random.shuffle(index)
    x_del = x_del[index]
    y_del = y_del[index]
    # print(x_del)
    # print(y_del)
    print('Down sample done!')
    return x_del, y_del


# summarize the number of rows with missing values for each column
def summarize_dataset(dataframe):
    for i in range(dataframe.shape[1]):
        # count number of rows with missing values
        n_miss = dataframe[[i]].isnull().sum()
        perc = n_miss / dataframe.shape[0] * 100
        print('> %d, Missing: %d (%.1f%%)' % (i, n_miss, perc))

## added by Annie Zhou on June 28th, 2022 1:05PM
def knn_imputation(X):

    print('Start KNN imputation ......')
    start = time() 

    imputer = KNNImputer(n_neighbors=5, weights='uniform', metric='nan_euclidean')
    # fit on the dataset
    print('Fitting the knn imputer .......')
    imputer.fit(X)
    Xtrans = imputer.transform(X)
    print('Imputation done!')

    print('Time taken for KNN imputation:' ,time() - start, 'seconds')
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

    return Xtrans
def mice(X):
    # lr = LinearRegression()
    # imp = IterativeImputer(estimator=lr,missing_values=np.nan, max_iter=10, verbose=2, imputation_order='roman',random_state=0)
    # X_trans=imp.fit_transform(X)


    imp = IterativeImputer(max_iter=10, random_state=0)
    # >imp.fit([[1, 2], [3, 6], [4, 8], [np.nan, 3], [7, np.nan]])
    imp.fit(X)  
    return imp.transform(X)


## modified by Annie Zhou on May 5th, 2022
if __name__ == "__main__":
    data_path =  os.getcwd() + "/merged/"
    train_nosepsis_full = np.load('train_nosepsis.npy')
    train_sepsis_full = np.load('train_sepsis.npy')
    

    ## list the columns that is missing > 50 and string columns
    cols_to_remove = []
    with open( os.getcwd() +'/col_to_remove.txt', 'r') as fp:
        for line in fp:
                # remove linebreak from a current name
                # linebreak is the last character of each line
            x = line[:-1]

                # add current item to the list
            cols_to_remove.append(x)
    cols_to_remove = list(set(cols_to_remove))
    print('cols to be removed:\n', cols_to_remove)
 
    train_nosepsis  = train_nosepsis_full[0:int(len(train_nosepsis_full)/100)] 
    train_sepsis  = train_sepsis_full[0:int(len(train_sepsis_full)/100)] 
    # 5-fold cross validation was implemented and five XGBoost models were produced
    kfold = KFold(n_splits=5, shuffle=True, random_state=np.random.seed(12306))
 








    # K-Folds cross-validator
    # Provides train/test indices to split data in train/test sets. Split dataset into k consecutive folds (without shuffling by default).
    # Each fold is then used once as a validation while the k - 1 remaining folds form the training set.


    for (k, (train0_index, val0_index)), (k, (train1_index, val1_index)) in \
            zip(enumerate(kfold.split(train_nosepsis)), enumerate(kfold.split(train_sepsis))):
        print('{}th fold **********************************'.format(k+1))
        # print(train0_index)
        # print(val0_index)

        # training set
        print('Training set processing .......')
        train_set = np.append(train_nosepsis[train0_index], train_sepsis[train1_index])
        np.random.shuffle(train_set)
        x_train, y_train = downsample(train_set, data_path,cols_to_remove)



        # x_train, y_train = data_process(train_set, data_path)
        y_train_nan_index = np.argwhere(np.isnan(y_train))
        # print('y_train_nan_index:',y_train_nan_index)
        y_train = np.delete(y_train, (y_train_nan_index), axis=0)
        x_train = np.delete(x_train, (y_train_nan_index), axis=0)
        print('Training dataset Missing: %d' % sum(np.isnan(x_train).flatten()))
        
        # x_train_trans = knn_imputation(x_train)
        x_train_trans = mice(x_train)
        print('x_train.shape', x_train.shape)
        print('x_train_trans.shape', x_train_trans.shape)

         
        # print(y_train)
        print('Validation set processing .......')
        # validation set
        val_set = np.append(train_nosepsis[val0_index], train_sepsis[val1_index])
        np.random.shuffle(val_set)
        # x_val, y_val = data_process(val_set, data_path) 
        x_val, y_val = downsample(val_set, data_path, cols_to_remove)
        y_val_nan_index = np.argwhere(np.isnan(y_val))
        # print('y_val_nan_index:',y_val_nan_index)
        y_val = np.delete(y_val, (y_val_nan_index), axis=0)
        x_val = np.delete(x_val, (y_val_nan_index), axis=0)

        print('Validation dataset Missing: %d' % sum(np.isnan(x_val).flatten()))
        # x_val_trans = knn_imputation(x_val)
        x_val_trans = mice(x_val)
        print('x_val.shape', x_val.shape)
        print('x_val_trans.shape', x_val_trans.shape)
        # print(y_val)
        train_model(k, x_train_trans, y_train, x_val_trans, y_val, save_model_dir = os.getcwd() + '/xgb_model/')
    
 