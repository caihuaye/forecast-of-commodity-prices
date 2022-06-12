import scipy.io as sio
import numpy as np
import datetime
# 去除不必要的警告信息
import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter(action='ignore', category=FutureWarning)

from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
from sklearn.model_selection import KFold

from hyperopt import fmin, tpe, hp,rand                         # 调参工具
from sklearn.preprocessing import MinMaxScaler  


# In[ ]:
def model_tuner(args):
    global X_train, X_test, Y_train, Y_test
    # load parameters

    ################# 要改模型改这里 #################
    para1 = args['learning_rate']
    para2 = args['n_estimators']
    lgbm = LGBMRegressor(num_leaves=31,learning_rate = para1, n_estimators = para2)
    #################################################

    # evaluate parameter with 5-fold cross validation
    mse_vals = []
    kf = KFold(n_splits = 5,shuffle = True)
    for temp_train_idx,temp_test_idx in kf.split(X_train):
        X_train_temp = X_train[temp_train_idx,:]
        Y_train_temp = Y_train[temp_train_idx]
        X_val = X_train[temp_test_idx,:]
        Y_val = Y_train[temp_test_idx]
        ################# 要改模型改这里 #################
        lgbm.fit(X_train_temp, Y_train_temp)
        res = lgbm.predict(X_val)
        #################################################
        mse_vals.append(mean_squared_error(Y_val, res,multioutput = 'uniform_average'))
    return np.mean(mse_vals)


# In[ ]:


def model_evaluation(data,turn,train_idxes,test_idxes):
    global maxeval, X_train, X_test, Y_train, Y_test

    # load data
    train_index = train_idxes[turn][0]
    test_index = test_idxes[turn][0]

    X_train = data['X'][train_index,:]
    Y_train = data['Y'][0,train_index]
    X_test = data['X'][test_index,:]
    Y_test = data['Y'][0,test_index]


    # search the optimal hyperparameter
    ## search space
    ################# 要改模型改这里 #################
    space = {
            'learning_rate': hp.choice('learning_rate',[10**i for i in range(-3,1)]),
            'n_estimators': hp.choice('n_estimators',[20*(i+1) for i in range(5)])
            }
    #################################################

    ## model tuner
    best = fmin(fn=model_tuner, space=space,algo=rand.suggest, max_evals=maxeval)


    # retrain model with the optimal parameter
    ################# 要改模型改这里 #################
    para_lr = [10**i for i in range(-3,1)]
    para_est = [20*(i+1) for i in range(5)]
    
    best_lr = para_lr[best['learning_rate']]
    best_est = para_est[best['n_estimators']]
    best_para = [best_lr,best_est]
    #################################################


    # test & evaluate the model
    ################# 要改模型改这里 #################
    lgbm = LGBMRegressor(num_leaves=31,learning_rate = best_lr, n_estimators = best_est)
    lgbm.fit(X_train,Y_train)

    res = lgbm.predict(X_test)
    fea_importance = lgbm.feature_importances_
    #################################################

    # calculate the metric
    mse_value = mean_squared_error(Y_test, res, multioutput = 'uniform_average')
    R_value, _ = pearsonr(Y_test, res)


    print('=========> Iter',turn,': Model testing complete.\n',
          'MSE:',mse_value,'Correlation Coefficience:',R_value,'\n')
    return best_para,res,Y_test,mse_value,R_value,fea_importance

# In[ ]: 
if __name__ == "__main__":
    global maxeval
    # parameter setting
    model_name = 'lgbm'
    maxeval = 10           # 调参工具每次最多搜索多少次
    kfold_num = 5           # cv fold number
    iter_time = 1           # random run times

    # load data
    idxes = sio.loadmat('index.mat')
    data = sio.loadmat('ori_data.mat')

    min_max_scaler = MinMaxScaler()
    data['X'] = min_max_scaler.fit_transform(data['X'])

    # go.
    for iter in range(iter_time):
        train_idxes = []
        test_idxes = []

        train_idxes = idxes['total_train_idxes'][iter]
        test_idxes = idxes['total_test_idxes'][iter]

        mse_values = []
        corr_values = []
        y_pred = []
        y_true = []
        best_paras = []
        fea_importances = []
        total_starttime = datetime.datetime.now()
        for i in range(kfold_num):
            starttime = datetime.datetime.now()
            best_para,res_score,target_score,mse_val,corr_value,fea_importance = model_evaluation(data,i,train_idxes,test_idxes)

            best_paras.append(best_para)
            y_pred.append(res_score)
            y_true.append(target_score)
            mse_values.append(mse_val)
            corr_values.append(corr_value)
            fea_importances.append(fea_importance)

            endtime = datetime.datetime.now()
            elapsed_sec = (endtime - starttime).total_seconds()
            saved_filename = './model/'+model_name+'_'+str(iter)+'.mat'
            sio.savemat(saved_filename, {'MSE':np.array(mse_values),
                                'R_value':np.array(corr_values),
                                'y_true':y_true,
                                'y_pred':y_pred,
                                'best_para':best_paras,
                                'fea_importances':fea_importances})

        total_endtime = datetime.datetime.now()
        total_elapsed_sec = (total_endtime - total_starttime).total_seconds()
        
        e = datetime.datetime.today()
        with open(f'./training_log/log_{model_name}_{e:%Y_%m%d_%H_%M_%S}.txt', 'w') as log:
            log.write(f'====================> {model_name} 第 {iter} 次随机运行结束\n')
            log.write(f'MSE: {np.mean(mse_values)} ± {np.std(mse_values)}\n')
            log.write(
                f'Correlation Coefficience: {np.mean(corr_values)} ± {np.std(corr_values)}\n')
            log.write(f'总用时: {total_elapsed_sec:.2f} 秒\n')
            log.close()
        
        print('====================>'+model_name+'第'+str(iter)+'次随机运行结束')
        print('MSE:',np.mean(mse_values),'±',np.std(mse_values))
        print('Correlation Coefficience:',np.mean(corr_values),'±',np.std(corr_values))
        print('\n总用时: '+ '{:.2f}'.format(total_elapsed_sec) + " 秒。")
