#111111
# coding=utf8
import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split,StratifiedKFold
from sklearn.metrics import f1_score
import xgboost as xgb
import datetime
import time
train_df = pd.read_csv('./data/train_all.csv')
test_df = pd.read_csv('./data/republish_test.csv')
print (train_df.shape)



train_df = train_df[train_df.gender != '\\N']
# test = test[test.gender != '\\N']
train_df['gender'] = train_df['gender'].apply(lambda x : int(x))
test_df['gender'] = test_df['gender'].apply(lambda x : int(x))

train_df = train_df[train_df.age != '\\N']
# test = test[test.age != '\\N']
train_df['age'] = train_df['age'].apply(lambda x : int(x))
test_df['age'] = test_df['age'].apply(lambda x : int(x))

train_df = train_df[train_df['2_total_fee'] != '\\N']
# test = test[test['2_total_fee'] != '\\N']
test_df.loc[test_df['2_total_fee'] == '\\N','2_total_fee'] = 0.0
train_df['2_total_fee'] = train_df['2_total_fee'].apply(lambda x : float(x))
test_df['2_total_fee'] = test_df['2_total_fee'].apply(lambda x : float(x))

train_df = train_df[train_df['3_total_fee'] != '\\N']
# test = test[test['3_total_fee'] != '\\N']
test_df.loc[test_df['3_total_fee'] == '\\N','3_total_fee'] = 0.0
train_df['3_total_fee'] = train_df['3_total_fee'].apply(lambda x : float(x))
test_df['3_total_fee'] = test_df['3_total_fee'].apply(lambda x : float(x))


label = train_df.pop('current_service')
le = LabelEncoder()
label = le.fit_transform(label)
#train_df['last_month_traffic'][train_df['service_type']==3] = train_df['last_month_traffic'][train_df['service_type']==3]/1024
#test_df['last_month_traffic'][test_df['service_type']==3] = test_df['last_month_traffic'][test_df['service_type']==3]/1024
test_df.loc[test_df['service_type']==3,'last_month_traffic']=test_df.loc[test_df['service_type']==3,'last_month_traffic']/1024
test_df.loc[test_df['service_type']==3,'service_type']=4
train_df['fun_total_fee'] = 0
train_df.loc[train_df['3_total_fee']<0,'fun_total_fee'] = 1
#train_df['fun_total_fee'][(train_df['3_total_fee']<0)] = 1
test_df['fun_total_fee'] = 0
test_df.loc[test_df['3_total_fee']<0,'fun_total_fee'] = 1
#test_df['fun_total_fee'][(test_df['3_total_fee']<0)] = 1
# train_df['fun_total_fee'] = 0
# train_df['fun_total_fee'][(train_df['4_total_fee']<0)] = 1
# test_df['fun_total_fee'] = 0
# test_df['fun_total_fee'][(test_df['4_total_fee']<0)] = 1
train_df['fun_last_month_traffic'] = 0
train_df.loc[train_df['last_month_traffic']==800,'fun_last_month_traffic'] = 1
#train_df['fun_last_month_traffic'][(train_df['last_month_traffic']==800)] = 1
test_df['fun_last_month_traffic'] = 0
test_df.loc[test_df['last_month_traffic']==800,'fun_last_month_traffic'] = 1
#test_df['fun_last_month_traffic'][(test_df['last_month_traffic']==800)] = 1

train_df['max_total_fee1'] = map(lambda x,y:max(x,y),train_df['1_total_fee'],train_df['2_total_fee'])
train_df['max_total_fee2'] = map(lambda x,y:max(x,y),train_df['max_total_fee1'],train_df['3_total_fee'])
train_df['max_total_fee3'] = map(lambda x,y:max(x,y),train_df['max_total_fee2'],train_df['4_total_fee'])
test_df['max_total_fee1'] = map(lambda x,y:max(x,y),test_df['1_total_fee'],test_df['2_total_fee'])
test_df['max_total_fee2'] = map(lambda x,y:max(x,y),test_df['max_total_fee1'],test_df['3_total_fee'])
test_df['max_total_fee3'] = map(lambda x,y:max(x,y),test_df['max_total_fee2'],test_df['4_total_fee'])
train_df['max_total_fee3']=pd.to_numeric(train_df['max_total_fee3'],errors='coerce')
test_df['max_total_fee3']=pd.to_numeric(test_df['max_total_fee3'],errors='coerce')
train_df.drop(['max_total_fee1','max_total_fee2'],axis=1,inplace=True)
test_df.drop(['max_total_fee1','max_total_fee2'],axis=1,inplace=True)

train_df['min_total_fee1'] = map(lambda x,y:min(x,y),train_df['1_total_fee'],train_df['2_total_fee'])
train_df['min_total_fee2'] = map(lambda x,y:min(x,y),train_df['min_total_fee1'],train_df['3_total_fee'])
train_df['min_total_fee3'] = map(lambda x,y:min(x,y),train_df['min_total_fee2'],train_df['4_total_fee'])
test_df['min_total_fee1'] = map(lambda x,y:min(x,y),test_df['1_total_fee'],test_df['2_total_fee'])
test_df['min_total_fee2'] = map(lambda x,y:min(x,y),test_df['min_total_fee1'],test_df['3_total_fee'])
test_df['min_total_fee3'] = map(lambda x,y:min(x,y),test_df['min_total_fee2'],test_df['4_total_fee'])
train_df['min_total_fee3']=pd.to_numeric(train_df['min_total_fee3'],errors='coerce')
test_df['min_total_fee3']=pd.to_numeric(test_df['min_total_fee3'],errors='coerce')
train_df.drop(['min_total_fee1','min_total_fee2'],axis=1,inplace=True)
test_df.drop(['min_total_fee1','min_total_fee2'],axis=1,inplace=True)

dummies_net_service = pd.get_dummies(train_df['net_service'],prefix='net_service')
dummies_is_mix_service = pd.get_dummies(train_df['is_mix_service'],prefix='is_mix_service')
dummies_many_over_bill = pd.get_dummies(train_df['many_over_bill'],prefix='many_over_bill')
#dummies_contract_type = pd.get_dummies(train_df['contract_type'],prefix='contract_type')
dummies_is_promise_low_consume = pd.get_dummies(train_df['is_promise_low_consume'],prefix='is_promise_low_consume')
dummies_gender = pd.get_dummies(train_df['gender'],prefix='gender')
dummies_complaint_level = pd.get_dummies(train_df['complaint_level'],prefix='complaint_level')
train_df = pd.concat([train_df,dummies_net_service,dummies_is_mix_service,dummies_many_over_bill],axis=1)
train_df = pd.concat([train_df,dummies_is_promise_low_consume,dummies_gender,dummies_complaint_level],axis=1)
train_df.drop(['net_service','is_mix_service','many_over_bill'],axis=1,inplace=True)
train_df.drop(['is_promise_low_consume','gender','complaint_level'],axis=1,inplace=True)

dummies_net_service = pd.get_dummies(test_df['net_service'],prefix='net_service')
dummies_is_mix_service = pd.get_dummies(test_df['is_mix_service'],prefix='is_mix_service')
dummies_many_over_bill = pd.get_dummies(test_df['many_over_bill'],prefix='many_over_bill')
#dummies_contract_type = pd.get_dummies(test_df['contract_type'],prefix='contract_type')
dummies_is_promise_low_consume = pd.get_dummies(test_df['is_promise_low_consume'],prefix='is_promise_low_consume')
dummies_gender = pd.get_dummies(test_df['gender'],prefix='gender')
dummies_complaint_level = pd.get_dummies(test_df['complaint_level'],prefix='complaint_level')
test_df = pd.concat([test_df,dummies_net_service,dummies_is_mix_service,dummies_many_over_bill],axis=1)
test_df = pd.concat([test_df,dummies_is_promise_low_consume,dummies_gender,dummies_complaint_level],axis=1)
test_df.drop(['net_service','is_mix_service','many_over_bill'],axis=1,inplace=True)
test_df.drop(['is_promise_low_consume','gender','complaint_level'],axis=1,inplace=True)

#new feature
#计算每个月的total-fee占总的total_fee的比例
train_df['sum_total_fee'] = (train_df['1_total_fee']+train_df['2_total_fee']+train_df['3_total_fee']+train_df['4_total_fee'])
test_df['sum_total_fee']= (test_df['1_total_fee']+test_df['2_total_fee']+test_df['3_total_fee']+test_df['4_total_fee'])
train_df['1_total_fee_rate']=(train_df['1_total_fee']/train_df['sum_total_fee'])
train_df['2_total_fee_rate']=(train_df['2_total_fee']/train_df['sum_total_fee'])
train_df['3_total_fee_rate']=(train_df['3_total_fee']/train_df['sum_total_fee'])
train_df['4_total_fee_rate']=(train_df['4_total_fee']/train_df['sum_total_fee'])
test_df['1_total_fee_rate']=(test_df['1_total_fee']/test_df['sum_total_fee'])
test_df['2_total_fee_rate']=(test_df['2_total_fee']/test_df['sum_total_fee'])
test_df['3_total_fee_rate']=(test_df['3_total_fee']/test_df['sum_total_fee'])
test_df['4_total_fee_rate']=(test_df['4_total_fee']/test_df['sum_total_fee'])
train_df.drop(['sum_total_fee'],axis=1,inplace=True)
test_df.drop(['sum_total_fee'],axis=1,inplace=True)

#计算四个月toatal-fee的众数

def mode(x1,x2,x3,x4):
    count_dict={}
    mode_x = np.array([x1,x2,x3,x4])
    for i in mode_x:
        if count_dict.has_key(i):
            count_dict[i]+=1
        else:
            count_dict[i]=1
    max_appear = 0
    for v in count_dict.values():
        if v>max_appear:
            max_appear=v;
    if max_appear==1:
        return x1;
    mode_list=[];
    for k,v in count_dict.items():
        if v==max_appear:
            return k
    
train_df['mode_total_fee'] = map(lambda x1,x2,x3,x4:mode(x1,x2,x3,x4),train_df['1_total_fee'],train_df['2_total_fee'],train_df['3_total_fee'],train_df['4_total_fee'])
test_df['mode_total_fee'] = map(lambda x1,x2,x3,x4:mode(x1,x2,x3,x4),test_df['1_total_fee'],test_df['2_total_fee'],test_df['3_total_fee'],test_df['4_total_fee'])

#month_traffic和local_traffic_month



feature = [value for value in train_df.columns.values if
                   value not in ['user_id']]

print(feature)
#y = train_df['current_service']
train_id = train_df['user_id']
X = train_df.drop(['user_id'],axis=1)
test_id = test_df['user_id']
X_test = test_df.drop(['user_id'],axis=1)
X,X_test = X.values,X_test.values

def XGB():
    clf = xgb.XGBClassifier(max_depth=12, learning_rate=0.05,
                            n_estimators=1200, silent=True,
                            objective="multi:softmax",
                            nthread=40, gamma=0,
                            max_delta_step=0, subsample=1, colsample_bytree=0.8, colsample_bylevel=0.8,
                            reg_alpha=0, reg_lambda=0.5, scale_pos_weight=1,
                            base_score=0.5, seed=42, missing=None,num_class=11)
    return clf

n_splits = 5
seed = 42
skf = StratifiedKFold(n_splits=n_splits,random_state=seed,shuffle=True)
online = True
cv_prediction = []
if online:
        print ('online')
        model = XGB()
        for index,(train_index,test_index) in enumerate(skf.split(X,label)):
            print(index)
            X_train,X_val,y_train,y_val = X[train_index],X[test_index],label[train_index],label[test_index]
            clf = model.fit(X_train,y_train,eval_set=[(X_val,y_val)],early_stopping_rounds=150,eval_metric = "mlogloss",verbose=1)
            #xx_pred = clf.predict(X_val,num_iteration=clf.best_iteration)
            y_test = clf.predict(X_test,ntree_limit=clf.best_ntree_limit)
            if index ==0:
                cv_prediction = np.array(y_test).reshape(-1,1)
            else:
                cv_prediction = np.hstack((cv_prediction,np.array(y_test).reshape(-1,1)))           
        #model.fit(train_df[feature], label, eval_set=[(train_df[feature], label)], eval_metric = "mlogloss",verbose=1)
        feature_list = model.feature_importances_
        now = datetime.datetime.now()
        now = now.strftime('%m-%d-%H-%M')
        pd.DataFrame({'feature':feature,
              'score':feature_list,}
            ).to_csv('./result/feature_importance_%s.csv'%now,index=False)
        #pred = model.predict(test_df[feature])
        result = []
        for prediction in cv_prediction:
            result.append(np.argmax(np.bincount(prediction)))  
        pred = le.inverse_transform(result)
        test_df['predict'] = pred
        test_df[['user_id', 'predict']].to_csv('./result/xgb_feature_skf_v1_%s.csv'%now, index=False)
