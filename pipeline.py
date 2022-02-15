import classifires as cl
import numpy as np
import pandas as pd
from sklearn import preprocessing
import sklearn.model_selection as ms
from sklearn.model_selection import cross_validate
from sklearn import linear_model
from sklearn import feature_selection as fs
import sklearn.decomposition as skde
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as ss
import math
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import LabelEncoder , OneHotEncoder
LE = LabelEncoder()
OHE = OneHotEncoder(handle_unknown='ignore')
import joblib
import ast
from sklearn.preprocessing import PolynomialFeatures
from sklearn.base import BaseEstimator, TransformerMixin
from imblearn.over_sampling import SMOTE
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import mean_squared_error
import math

#------read & view-----
def read_data(filepath):
    df = pd.read_csv(filepath)
    return df


def missing_value_analysis(df,drop_th):
    samples = len(df); dropped_col=[]
    dfc=df.copy(); dfc=dfc.dropna()
    missing_dict = {}
    for c in df.columns:
        n_missing = df[c].isnull().sum()
        missing_ratio = n_missing/samples
#        x_scores=feature_scores(dfc.drop([c],axis=1), dfc[c]
        
        if missing_ratio >= drop_th:
            df = df.drop([c], axis=1)
            dropped_col.append(c)
        elif n_missing > 0:
            missing_dict[c] = []
            missing_dict[c].append((n_missing, missing_ratio, df[c].dtype))
            
    print("dropped_columns-->", dropped_col)
    print("other missing colums-->")
    for k,v in missing_dict.items():
        print(k, "=" , v); print("")
    return df
    

def correlations(x, th=0.5):
   cors = x.corr()
   print(cors[abs(cors)>th]); print("")



#--------feature analysis----------#
def sort_by_value(dictx):
    dicty=sorted(dictx.items(),key=lambda item: item[1],reverse=True)
    return dicty


def data_to_df(x):
    if not isinstance(x, pd.DataFrame):
        x = pd.DataFrame(x)
    return x


def is_categorical(data):
    uv = len(list(set(data))); #print("unique-->", uv)
    if (uv/len(data) <= 0.05 or uv <= 10) and type(data[0]) != float:
        return 1
    else:
        return 0


def assert_to_discrete(data):
    data = np.array(data)
    if max(data)<10:
        try:
            data = data*10;
        except:
            pass
    if is_categorical(data) == 1:
        cat_data = data.astype('str')
    else:
        s = int(min(data))
        e = int(max(data))
        itr = int(np.std(data))
        if itr<1:
            itr = int(np.mean(data))
        bins = []; #print("itr-------------->",itr)
        for i in range(s,e,itr):
            bins.append(i)
        bins.append(e + itr)
        bins.append(bins[-1] + itr); #buffer

        bin_range = {}
        for i in range(len(bins)-1):
            bin_range[i]=[bins[i],bins[i+1]]

        cat_data=[]; missed=[]
        for v in data:
            f=0
            for i, br in bin_range.items():
                if v >= br[0] and v < br[1]:
                    f=1
                    # print("c1", v, br, "--->", i)
                    cat_data.append(i); break
            if f==0: missed.append(v)

    # print("bin range---> ", s, e, itr)
    # print("bins--->", bins)
    # print("bin dict--->", bin_range)
    # print("missed to bin--->", missed)
    # print('counter--->', token_counter(cat_data))
    return cat_data


def feature_analysis(x,y):
    x = data_to_df(x)
    y_type = y.dtype;  # original dtype
    x=x.dropna()
    #by default add 'ybin' as categorical
    if y_type != 'O':
        x['ybin'] = assert_to_discrete(y)
    else:
        x['ybin'] = y

    label_count = len(x['ybin'].unique());
    # print('label-->', label_count)

    feature_scores={}
    for col in x.columns:
        if col == 'ybin': continue
        x_type = x[col].dtype
        if x_type != 'O':
            x[col] = assert_to_discrete(x[col])

        gx = x.groupby([col,'ybin'])['ybin'].count(); #print(gx)
        gxr = gx/len(x)
        fgx = gxr[gxr.values >= 0]; #print(fgx)
        ix = fgx.index
        clist={}
        for c, yb in ix:
            if c not in clist: clist[c] = []
            clist[c].append(yb)
        availability = len(clist)/len(x[col].unique())
        penalty=[]
        for k, v in clist.items():
            penalty.append(1/len(v))
        avg_penalty = np.mean(np.array(penalty))
        ns = availability*avg_penalty
        feature_scores[col]=ns
    feature_scores = sort_by_value(feature_scores)
    return feature_scores



#-----feature engineering + selection + reduction------

def feature_selection(X,Y):
    X, Y, mmd = auto_transform_data(X,Y)
    feature_folds = ms.KFold(n_splits=10, shuffle = True)
    #customize parameter if needed
    logistic_mod = linear_model.LogisticRegression(C = 10)
    selector = fs.RFECV(estimator = logistic_mod, cv = feature_folds,
                      scoring = 'roc_auc')
    selector = selector.fit(X, Y)
    r = selector.ranking_
    c = X.columns
    ranks={}
    for i in range(len(r)):
        ranks[c[i]]=r[i]
    return ranks
    

def polynomial_features(x,degree):
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    i=0
    for c in x.columns:
        if x[c].dtype != 'O':
            i=i+1
            poly_feat = poly.fit_transform(x[c].reshape(-1,1)); #print(poly_feat[0])
            poly_df = pd.DataFrame(data=poly_feat, columns = ["f"+str(i)+str(d) for d in range(degree)])
            x = x.drop([c],axis=1)
            x=x.join(poly_df)
    return x


class GaussianFeatures(BaseEstimator, TransformerMixin):
    """Uniformly spaced Gaussian features for one-dimensional input"""
    
    def __init__(self, N, width_factor=2.0):
        self.N = N
        self.width_factor = width_factor
    
    @staticmethod
    def _gauss_basis(x, y, width, axis=None):
        arg = (x - y) / width
        return np.exp(-0.5 * np.sum(arg ** 2, axis))
        
    def fit(self, X, y=None):
        # create N centers spread along the data range
        self.centers_ = np.linspace(X.min(), X.max(), self.N)
        self.width_ = self.width_factor * (self.centers_[1] - self.centers_[0])
        return self
        
    def transform(self, X):
        return self._gauss_basis(X[:, :, np.newaxis], self.centers_,
                                 self.width_, axis=1)


def gaussian_features(x,y,n_dim):
    i=0
    for c in x.columns:
        if x[c].dtype != 'O':
            i=i+1
            gaus_fit = GaussianFeatures(n_dim).fit(x[c],y)
            gaus_feat = gaus_fit.transform(x[c].reshape(-1,1))
            gaus_df = pd.DataFrame(data=gaus_feat, columns = ["f"+str(i)+str(d) for d in range(n_dim)])
            x = x.drop([c],axis=1)
            x=x.join(gaus_df,lsuffix='_left', rsuffix='_right')
    return x


def oversampling(x,y):
    smote = SMOTE()
    x,y = smote.fit_resample(x,y)
    return x,y


def matrix_correction(x, y):
    x = pd.DataFrame(data=x, columns=[str(i) for i in range(x.shape[1])])
    indices_to_keep = ~x.isin([np.nan, np.inf, -np.inf]).any(1); 
    print("indices-->",indices_to_keep.value_counts())
    x = x[indices_to_keep]
    y = y[indices_to_keep]
    return x,y


#-----transformations-------
    
def skew_correction(X):
    for col in X.columns:
        if X[col].dtype != 'O':
            skewval = X[col].skew()
            if abs(skewval) > 0.7:
                X[col] = np.log1p(X[col])
    return X


def max_normalization(X):
    feature_max=[]
    for col in X.columns:
        if X[col].dtype != 'O':
            fmax = max(X[col])
            X[col] = X[col]/fmax
            feature_max.append(fmax)
    with open("models/feature_max.txt","w") as f:
        f.write(str(feature_max))
    return X, feature_max


def minmax_normalization(X):
    scaler = preprocessing.MinMaxScaler().fit(X)
    X = scaler.transform(X)
    joblib.dump(scaler,"models/norm_scaler.pkl")
    return X


def Standardization(X):
    scaler = preprocessing.StandardScaler().fit(X)
    X = scaler.transform(X)
    joblib.dump(scaler,"models/std_scaler.pkl")
    return X


def split_num_cat(X):
    x_num = X.select_dtypes(exclude=['O'])
    x_cat = X.select_dtypes(include=['O'])
    return x_num, x_cat


def join_num_cat(x_num, x_cat):
    X = np.concatenate((x_num,x_cat),axis=1)
    X = pd.DataFrame(X)
    return X


def label_encode(X):
    for col in X.columns:
        if X[col].dtype == 'O':
            labels = LE.fit_transform(X[col].values); print(labels.shape)
            X[col] = labels
    return X


def onehot_encode(X_cat):
    print(X_cat.shape)
    X_cat_le = X_cat.apply(lambda col: LE.fit_transform(col.reshape(1, -1))); #print(S)
    OHE_model = OHE.fit(X_cat_le)
    joblib.dump(OHE_model,"models/ohe_model.pkl")
    X_cat_ohe = OHE.transform(X_cat_le).toarray();  #print(2,F.shape, F[0])
    return X_cat_ohe
            

def auto_transform_data(X, Y):
    #log transform, one-hot encoding, vectorization done - yet to normalize
    NewX, NewY, model_meta_data = cl.prepare_data(X, Y)
    return NewX, NewY, model_meta_data


def reduce_dimensions(X, n_dim):
    pca_model = skde.PCA(n_components=n_dim)
    pca_fit = pca_model.fit(X)
    X = pca_fit.transform(X)
    print("PCA variance:")
    print(pca_fit.explained_variance_ratio_); print("")
    joblib.dump(pca_fit, "models/pca_model.pkl")
    return X

#-----validations-----

def compare_models(x, y):
    mmd, NewX = cl.compare_models(x,y)
    for k, v in mmd.items():
        print(k, ":", v)
    print(NewX.shape)
    return mmd, NewX
    

def kfold_cross_validate(model,X,Y,rstate):
    feature_folds = ms.KFold(n_splits=10, shuffle = True, random_state=rstate)
    cv_estimate = ms.cross_val_score(model, X, Y, cv = feature_folds)
    print('Mean performance metric = ', np.mean(cv_estimate))
    print('SDT of the metric       = ', np.std(cv_estimate))
    print('Outcomes by cv fold')
    for i, x in enumerate(cv_estimate):
        print('Fold ', (i+1, x))
    print("")
  

def select_best_parameters(model, param_grid, X, Y,rstate):
    feature_folds = ms.KFold(n_splits=10, shuffle = True,random_state=rstate)
    clf = ms.GridSearchCV(estimator = model, 
                          param_grid = param_grid, 
                          cv = feature_folds,
                          scoring = 'roc_auc',
                          return_train_score = True)
    clf.fit(X, Y)
    print(clf.best_estimator_, clf.best_score_)
    return clf.best_estimator_


def clf_train_test(model, x, y, rstate, model_id):
    print(model); print("")
    x_train,x_test,y_train,y_test=ms.train_test_split(x,y,test_size=0.2, random_state=rstate)
    model_fit = model.fit(x_train,y_train)
    joblib.dump(model_fit, "models/model_"+str(model_id))
    train_pred = model_fit.predict(x_train)
    test_pred = model_fit.predict(x_test)
    print("Training:")
    print(classification_report(y_train, train_pred))
    print("roc_auc:", roc_auc_score(y_train, train_pred))
    print("")
    print("Testing:")
    print(classification_report(y_test, test_pred))
    print("roc_auc:", roc_auc_score(y_test, test_pred))
    
    
def reg_train_test(model, x, y, rstate, model_id):
    print(model); print("")
    x_train,x_test,y_train,y_test=ms.train_test_split(x,y,test_size=0.2, random_state=rstate)
    model_fit = model.fit(x_train, y_train)
    joblib.dump(model_fit, "models/model_"+str(model_id))
    y_pred=model.predict(x_test)
#    y_test=np.expm1(y_test)
#    y_pred=np.expm1(y_pred)
    plt.scatter(y_test,y_pred)
    plt.grid()
    plt.xlabel('Actual y')
    plt.ylabel('Predicted y')
    plt.title('actual y vs predicted y')
    plt.show()
    rmse=math.sqrt(mean_squared_error(y_test, y_pred)); print("RMSE = ", rmse)
#    den = np.expm1(y)
    den = y
    er=rmse/np.mean(den); print("Error Rate = ", round(er*100,2),"%")
    eth = er*1.1
    diff = abs(y_test-y_pred).values
    best = diff[diff<eth]
    bp=len(best)/len(diff); print("Prediction Rate = ", round(bp*100,2),"%")

    