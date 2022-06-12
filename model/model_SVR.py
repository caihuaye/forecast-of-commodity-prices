from hyperopt import fmin, tpe, hp, rand                         # 调参工具
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
import scipy.io as sio
import numpy as np
import datetime
# 去除不必要的警告信息
import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter(action='ignore', category=FutureWarning)


# In[ ]:
def model_tuner(args):
    global X_train, X_test, Y_train, Y_test
    # load parameters

    ################# 要改模型改这里 #################
    para_kernel = 'linear'
    para1 = args['C']
    para3 = args['epsilon']
    svr = SVR(kernel=para_kernel, C=para1, epsilon=para3)
    #################################################

    # evaluate parameter with 5-fold cross validation
    mse_vals = []
    kf = KFold(n_splits=5, shuffle=True)
    for temp_train_idx, temp_test_idx in kf.split(X_train):
        X_train_temp = X_train[temp_train_idx, :]
        Y_train_temp = Y_train[temp_train_idx]
        X_val = X_train[temp_test_idx, :]
        Y_val = Y_train[temp_test_idx]
        ################# 要改模型改这里 #################
        svr.fit(X_train_temp, Y_train_temp)
        res = svr.predict(X_val)
        #################################################
        mse_vals.append(mean_squared_error(
            Y_val, res, multioutput='uniform_average'))
    return np.mean(mse_vals)


# In[ ]:


def model_evaluation(data, turn, train_idxes, test_idxes):
    global maxeval, X_train, X_test, Y_train, Y_test

    # load data
    train_index = train_idxes[turn][0]
    test_index = test_idxes[turn][0]

    X_train = data['X'][train_index, :]
    Y_train = data['Y'][0, train_index]
    X_test = data['X'][test_index, :]
    Y_test = data['Y'][0, test_index]

    # search the optimal hyperparameter
    # search space
    ################# 要改模型改这里 #################
    space = {
        'C': hp.choice('C', [1e-2, 1e-1, 1e0, 1e1, 1e2]),
        'epsilon': hp.choice('epsilon', [1e-2, 1e-1, 1e0])
    }
    #################################################

    # model tuner
    best = fmin(fn=model_tuner, space=space,
                algo=rand.suggest, max_evals=maxeval)

    # retrain model with the optimal parameter
    ################# 要改模型改这里 #################
    para_C = [1e-2, 1e-1, 1e0, 1e1, 1e2]
    para_eps = [1e-2, 1e-1, 1e0]

    best_C = para_C[best['C']]
    best_eps = para_eps[best['epsilon']]
    best_para = [best_C, best_eps]
    #################################################

    # test & evaluate the model
    ################# 要改模型改这里 #################
    svr = SVR(kernel='linear', C=best_C, epsilon=best_eps)
    svr.fit(X_train, Y_train)

    res = svr.predict(X_test)
    feature_importance = np.squeeze(svr.coef_)
    #################################################

    # calculate the metric
    mse_value = mean_squared_error(Y_test, res, multioutput='uniform_average')
    R_value, _ = pearsonr(Y_test, res)

    print('=========> Iter', turn, ': Model testing complete.\n',
          'MSE:', mse_value, 'Correlation Coefficience:', R_value, '\n')
    return best_para, res, Y_test, mse_value, R_value, feature_importance


# In[ ]:
if __name__ == "__main__":
    global maxeval
    # parameter setting
    model_name = 'SVR'
    maxeval = 10            # 调参工具每次最多搜索多少次
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
            best_para, res_score, target_score, mse_val, corr_value, fea_importance = model_evaluation(
                data, i, train_idxes, test_idxes)

            best_paras.append(best_para)
            y_pred.append(res_score)
            y_true.append(target_score)
            mse_values.append(mse_val)
            corr_values.append(corr_value)
            fea_importances.append(fea_importance)

            endtime = datetime.datetime.now()
            elapsed_sec = (endtime - starttime).total_seconds()
            saved_filename = './model/'+model_name+'_'+str(iter)+'.mat'
            sio.savemat(saved_filename, {'MSE': np.array(mse_values),
                                         'R_value': np.array(corr_values),
                                         'y_true': y_true,
                                         'y_pred': y_pred,
                                         'best_para': best_paras,
                                         'fea_importances': fea_importances})

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
        print('MSE:', np.mean(mse_values), '±', np.std(mse_values))
        print('Correlation Coefficience:', np.mean(
            corr_values), '±', np.std(corr_values))
        print('\n总用时: ' + '{:.2f}'.format(total_elapsed_sec) + " 秒。")
