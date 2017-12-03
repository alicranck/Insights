from config import *
import numpy as np
from sklearn.cluster import KMeans
import datetime
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
import sklearn.preprocessing as preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score
from operator import itemgetter
import xgboost as xgb
import pymongo
from pymongo import MongoClient
import pickle

client = MongoClient()
db=client.icu_db


#build and save clustering model
def make_clusters(n,df):
    #df=pd.DataFrame(list(db.all_feats_collection.find()))
    df=df[df['dayNum']==1]
    #df=df.drop(['temp_adm_num','isMortDay'],axis=1)
    df=df.drop(['temp_adm_num','APACHE II'],axis=1,errors='ignore')
    df=df.drop([x for x in df.columns if 'KCAL' in x or 'mortality' in x or 'cluster' in x],axis=1)
    df['chronic-category']=(df['chronic-category']>0).astype(int)
    #X=np.array(df)
    df=df.drop([x for x in df.columns if 'MortDay' in x or '_id' in x or 'Unnamed' in x or 'dayNum' in x or 'justification' in x],axis=1) #or 'sepsis' in x
    #print df.columns
    pr=[x for x in df.columns if 'PROTEIN' in x][0]
    ag=[x for x in df.columns if 'age' in x][0]
    gnd=[x for x in df.columns if 'gender' in x][0]
    sps=[x for x in df.columns if 'sepsis' in x][0]
    proteins=np.array(df.pop(pr))
    age=np.array(df.pop(ag))
    gender=np.array(df.pop(gnd))
    sepsis=np.array(df.pop(sps))
    X=np.array(df)

    label_sum=np.zeros(n)
    m_sum=np.zeros(n)
    m=np.zeros(len(df))
    
    m_index = -1
    for x in range(len(df.columns)):
        if 'mort_adm' in df.columns[x]:
            m_index = x

    print (m_index)
    for i in range(len(df)):
        if X[i][m_index]==1:
            m[i]+=1

    df=df.drop('mort_adm',axis=1)
    X=np.array(df)
    scaler=sklearn.preprocessing.StandardScaler()
    X=scaler.fit_transform(X)

    kmeans=KMeans(n_clusters=n,random_state=0).fit(X)
    l=kmeans.labels_
    c=kmeans.cluster_centers_
    ndf=pd.DataFrame(columns=['rows'])
    unscaledX=np.array(df)
    cluster_list=[[] for i in range(n)]
    pr_list,gender_list,age_list,sepsis_list  =np.zeros(n),np.zeros(n),np.zeros(n),np.zeros(n)
    props=[]
    
    
    for j in range(n):
        for i in range(len(df)):
            if l[i]==j:
                label_sum[j]+=1
                if m[i]==1:
                    m_sum[j]+=1
                pr_list[j]+=proteins[i]
                age_list[j]+=age[i]
                gender_list[j]+=gender[i]
                cluster_list[j].append(unscaledX[i])
                sepsis_list[j]+=sepsis[i]

    print (sum(m_sum))
    m_sum=np.divide(m_sum,label_sum)
    label_per=label_sum/float(len(l))*100
    for i in range(n):
        pr_list[i]=pr_list[i]/float(label_sum[i])
        gender_list[i]=gender_list[i]/float(label_sum[i])
        age_list[i]=age_list[i]/float(label_sum[i])
        sepsis_list[i]=sepsis_list[i]/float(label_sum[i])

    ndf['rows']=list(df.columns)+['sepsis','age','gender','protein_mean','% of patients','num of patients','% mortality']

    i=0
    for cluster in c:
        ndf['cluster'+str(i)]=1
        ndf['cluster'+str(i)]=list(cluster)+[sepsis_list[i]]+[age_list[i]]+[gender_list[i]]+[pr_list[i]]+[label_per[i]]+[label_sum[i]]+[m_sum[i]]
        i+=1
    from scipy import stats
    for i in range(n):
        k=0
        for j in range(n):
            if j!=i:
                if k==0:
                    rest_data=np.array(cluster_list[j])
                    print (rest_data.shape)
                    k=1
                else:
                    rest_data=np.concatenate((rest_data,np.array(cluster_list[j])))
                    #print rest_data.shape
        ttest= stats.ttest_ind(np.array(cluster_list[i]),rest_data)
        pvalues=ttest[1]
        print ("cluster"+str(i))
        cluster_features=[]
        for p in range(len(pvalues)):
            if pvalues[p]<0.025:
                cluster_features.append((df.columns[p],ttest[0][p],np.abs(ttest[0][p])))
        cluster_features=sorted(cluster_features,key=itemgetter(2))[-10:]
        positive, negative = [],[]
        cur_props={}
        for f in cluster_features:
            if f[1]>0:
                positive.append(f)
            else:
                negative.append(f)
            #print f
        cur_props['positive'] = positive
        cur_props['negative'] = negative
        #print cur_props
        props.append(cur_props)
    df_cols = ndf.columns
    ndf=ndf.append(pd.DataFrame([['main features']+props],columns=df_cols))
    return ndf, kmeans

def build_clustering_model(df):
	ndf,model=make_clusters(6,df)
	print(ndf)
	f=open(r''+projectPath+'\\backend\models\clustering_model.model','wb')
	pickle.dump(model,f)
	f.close()
	import datetime
	properties=ndf.to_dict('records')
	clustering_model={"name":"clustering_model",
				  "last_trained": str(datetime.datetime.now()),
				  "path": r''+projectPath+'\\backend\models\\clustering_model.model',
				  "properties": properties
				 }
	db.models_collection.insert_one(clustering_model)

###############################################################################################


# run and save mortality model
#client = MongoClient()
#db=client.icu_db
def build_mortality_model(df):
	df = df.drop([x for x in df.columns if ('_id' in x or 'MortDay' in x or 'Unnamed' in x or
                                            'PROTEINI' in x or 'protein' in x or 'APACHE' in x)],axis=1)
	df = df.drop([x for x in df.columns if 'mortality' in x or 'cluster' in x or 'justification' in x], axis=1)

	df['chronic-category']=(df['chronic-category']>0).astype(int)
	df=df[df['dayNum']==1]
	df['sepsis-categorical']=df['sepsis-categorical'].apply(lambda x: 1 if x>=1 else 0)
	train = df.sample(frac=0.8, random_state=200)
	test = df.drop(train.index)

	y_train,y_test=train.pop('mort_adm'), test.pop('mort_adm')

	from sklearn import preprocessing
	scaler=preprocessing.MinMaxScaler()

	train,test=train.drop(['temp_adm_num','APACHE II'],axis=1,errors='ignore'), test.drop(['temp_adm_num','APACHE II'],axis=1,errors='ignore')
	print (train.shape,test.shape)

	xgdmat = xgb.DMatrix(train, y_train)
	our_params = {'eta': 0.01, 'seed':0, 'subsample': 0.8, 'colsample_bytree': 0.8, 
			 'objective': 'binary:logistic', 'max_depth':7, 'min_child_weight':1} 
	final_gb = xgb.train(our_params, xgdmat, num_boost_round = 1200)
	testdmat = xgb.DMatrix(test)
	y_pred = final_gb.predict(testdmat)

	roc2=roc_auc_score(y_test,y_pred)
	from sklearn.metrics import roc_curve
	fpr, tpr, _ = roc_curve(y_test, y_pred)
	plt.figure()
	lw=2
	plt.plot(fpr, tpr, color='darkorange',
			 lw=lw, label='ROC curve (area = %0.2f)' % roc2)
	plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	#plt.title('Sepsis prediction based on first day')
	plt.legend(loc="lower right")
	plt.savefig(r''+projectPath+'\\backend\models\\mortality_roc_curve.png')
	pickle.dump(final_gb, open(r''+projectPath+'\\backend\models\\mortality_model.model', "wb"))
	xgb.plot_importance(final_gb).figure.savefig(r''+projectPath+'\\backend\models\\mortality_feature_importance.png')
	y_pred_binary=y_pred
	y_pred_binary[y_pred_binary > 0.5] = 1
	y_pred_binary[y_pred_binary <= 0.5] = 0
	acc=accuracy_score(y_pred_binary, y_test)
	import datetime
	mortality_model={"name":"mortality_model",
				  "accuracy": acc,
				  "auc": roc2,
				  "last_trained": str(datetime.datetime.now()),
				  "path": r''+projectPath+'\\backend\models\\mortality_model.model',
				  "features_importance_path": r''+projectPath+'\\backend\models\\mortality_feature_importance.png',
				  "roc_auc_curve_path": r''+projectPath+'\\backend\models\\mortality_roc_curve.png'
				 }
	db.models_collection.insert_one(mortality_model)

##########################################################################################################

###predict mortality X days before mortality###
def build_3daysForward_model(df3):
	ndf=pd.DataFrame(columns=['days','AUC_xgboost'])
	df3=df3.drop([x for x in df3.columns if ('mort_adm' in x or '_id' in x or 'Unnamed' in x or 'PROTEINI' in x or 'protein' in x or 'APACHE' in x or 'sepsis-categorical' in x)],axis=1)#'APACHE' in x or
	df3 = df3.drop([x for x in df3.columns if 'mortality' in x or 'cluster' in x or 'justification' in x], axis=1)
	df3['chronic-category']=(df3['chronic-category']>0).astype(int)
	for i in range(1,21):
		df=df3.drop([x for x in df3.columns if 'MortDay' in x and 'In'+str(i)+'Days' not in x],axis=1)
		train,test=df[df['temp_adm_num']<2013130000], df[df['temp_adm_num']>=2013130000]
		train,test=train.drop(['temp_adm_num'],axis=1), test.drop(['temp_adm_num'],axis=1)
		y_train,y_test=train.pop('isIn'+str(i)+'DaysMortDay'), test.pop('isIn'+str(i)+'DaysMortDay')

		xgdmat = xgb.DMatrix(train, y_train)
		our_params = {'eta': 0.1, 'seed':0, 'subsample': 0.8, 'colsample_bytree': 0.8, 
				 'objective': 'binary:logistic', 'max_depth':5, 'min_child_weight':1} 
		final_gb = xgb.train(our_params, xgdmat, num_boost_round = 600)
		if i==3:
			pickle.dump(final_gb, open(r''+projectPath+'\\backend\models\\3days_mortality_model.model', "wb"))
		testdmat = xgb.DMatrix(test)
		y_pred = final_gb.predict(testdmat)
		roc2=roc_auc_score(y_test,y_pred)
		
		ndf=ndf.append(pd.DataFrame([[i,roc2]],columns=['days','AUC_xgboost']))
		

	plt.plot(ndf['days'],ndf['AUC_xgboost'])
	import datetime
	daysForward_model={"name":"daysForward_model",
				  "last_trained": str(datetime.datetime.now()),
				  "days": list(ndf["days"]), #to later plot the graph using plt.plot(days,AUCs)
				  "AUCs": list(ndf['AUC_xgboost']),
				  "path": r''+projectPath+'\\backend\models\\3days_mortality_model.model'
				 }
	db.models_collection.insert_one(daysForward_model)


#################################################################################################

# linear regression on xgb results for justification

def build_xgb_outputs(df):
	df = df.drop([x for x in df.columns if (
	'justification' in x or '_id' in x or 'APACHE' in x or 'cluster' in x or 'next3days_mortality_prediction' in x or 'mortality_prediction' in x or 'MortDay' in x
	or 'Unnamed' in x or 'sepsis' in x or 'temp_adm_num' in x)], axis=1)
	df = df[df['dayNum'] == 1]
	yd = df.pop('mort_adm')
	xdf, ydf = pd.DataFrame(columns=df.columns), pd.DataFrame(columns=["y"])
	X = np.array(df)
	y = np.array(yd)
	print(len(df))

	for i in range(10):
		X_train, X_test, y_train, y_test = np.concatenate((X[:int(len(df) / 10 * i)], X[int(len(df) / 10 * (i + 1)):])),\
										   X[int(len(df) / 10 * i):int(len(df) / 10 * (i + 1))], np.concatenate((y[:int(len(df) / 10 * i)]
											,y[int(len(df) / 10 * (i + 1)):])), y[int(len(df) / 10 * i):int(len(df) / 10 * (i + 1))]
		print(len(X_train), len(X_test))
		xgdmat = xgb.DMatrix(X_train, y_train)
		our_params = {'eta': 0.1, 'seed': 0, 'subsample': 0.8, 'colsample_bytree': 0.8,
					  'objective': 'binary:logistic', 'max_depth': 7, 'min_child_weight': 1}
		final_gb = xgb.train(our_params, xgdmat, num_boost_round=1200)
		testdmat = xgb.DMatrix(X_test)
		y_pred = final_gb.predict(testdmat)
		roc2 = roc_auc_score(y_test, y_pred)
		print(roc2)

		tempx = pd.DataFrame(columns=df.columns, data=X_test)
		xdf = xdf.append(tempx)
		tempy = pd.DataFrame()
		tempy["y"] = y_pred
		ydf = ydf.append(tempy)

	return xdf, ydf


def get_lr_params(xdf, ydf):
	X, y = xdf.copy(), ydf.copy()
	scaler = preprocessing.StandardScaler()
	X = scaler.fit_transform(X)

	from sklearn import linear_model, cross_validation

	coeffs = []
	intercepts = []
	max_var, max_index = 0, 0

	for i in range(10):
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

		y_pred_train = y_train
		y_pred_test = y_test

		test_lr = np.array(X_test)
		train_lr = np.array(X_train)

		lr = linear_model.LinearRegression()

		lr.fit(train_lr, y_pred_train)
		coeffs.append(lr.coef_)
		intercepts.append(lr.intercept_)
		lr_pred = lr.predict(test_lr)
		y_pred = y_pred_test
		if lr_pred[np.argmax(lr_pred)] > 20:
			max_delete = np.argmax(lr_pred)
			# print "deleted value",lr_pred[max_delete],"at index",max_delete
			# print "len y_pred_test",len(y_pred_test), len(lr_pred),len(y_pred_train)
			lr_pred = np.delete(lr_pred, max_delete)
			y_pred = np.delete(np.array(y_pred_test), max_delete)
			test_lr = np.delete(test_lr, max_delete, 0)
		# The mean squared error
		# print("Mean squared error: %.2f"
		# % np.mean((lr_pred - y_pred) ** 2))
		# Explained variance score: 1 is perfect prediction
		# print('Variance score: %.2f' % lr.score(test_lr, y_pred))
		if lr.score(test_lr, y_pred) > max_var:
			max_var = lr.score(test_lr, y_pred)
			max_index = i
	return coeffs[max_index], intercepts[max_index]


def build_lr_model(df):
	xdf, ydf = build_xgb_outputs(df)
	coeffs, intercept = get_lr_params(xdf, ydf)
	coeffs = list(coeffs[0])
	intercept = list(intercept)
	# print coeffs, intercept
	linearRegression_model = {"name": "linearRegression_model",
							 "last_trained": str(datetime.datetime.now()),
							 "coeffs": coeffs,
							 "intercept": intercept
							 }
	db.models_collection.insert_one(linearRegression_model)