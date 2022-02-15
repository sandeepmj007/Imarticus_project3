import joblib, sklearn
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder , OneHotEncoder
LE = LabelEncoder()
from sklearn import model_selection
from sklearn.utils import shuffle
from sklearn.metrics import classification_report, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

######----default name for label is "class"--------#####

def sort_by_value(dictx):
    dicty=sorted(dictx.items(),key=lambda item: item[1],reverse=True)
    return dicty


def data_to_df(x):
    if not isinstance(x, pd.DataFrame):
        x = pd.DataFrame(x)
    return x

def clean_dataset(df):
    df.dropna(inplace=True)
    indices_to_keep_x = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    #indices_to_keep_y = ~NewY.isin([np.nan, np.inf, -np.inf]).any(1)
    return df[indices_to_keep_x]#,NewY[indices_to_keep_y]
    

def tex2vec(X):
    vectorizer = TfidfVectorizer(min_df=0.001, max_df=1.0, stop_words='english')
    X = X.values.astype('U')
    xlist = [t.lower() for t in X]
    vectors = vectorizer.fit(xlist)
    vectors = vectorizer.transform(xlist)
    vectors = vectors.toarray()
    return vectors, vectorizer
    

 
def get_models():
    models = {}
    models['LogR'] = LogisticRegression()
    models['KNN'] = KNeighborsClassifier()
    models['DTC'] = DecisionTreeClassifier()
    models['NBC'] = GaussianNB()
    models['SVC'] = SVC()
    models['RFC'] = RandomForestClassifier()
    models['GBC'] = GradientBoostingClassifier()
    models['MLP'] = MLPClassifier()
    return models


def prepare_data(X,Y):
    model_meta_data={'vectorized':[], 'one-hot_encoded':[], 'log_transformed':[]}
    n = 100
    #X = shuffle(X)
    X = data_to_df(X)
    X = clean_dataset(X)
    NewX = np.array([[0] for i in range(len(X))]);  # print(NewX.shape)
    for col in X.columns:
        # print(col)
        if X[col].dtype == 'O':
            sample = X[col][:n]
            tokens = [len(t.split()) for t in sample]
            avg_tokens = np.mean(np.array(tokens))
            cat_ratio = len(sample.unique()) / n
            if cat_ratio > 0.3 and avg_tokens >= 3:
                F, vec = tex2vec(X[col]);  # print(1,F.shape, F[0])
                NewX = np.concatenate((NewX, F), axis=1)
                model_meta_data['vectorized'].append(col)
            else:
                enc = OneHotEncoder(handle_unknown='ignore')
                S = X[col].values.reshape(1, -1)
                ohe_model = enc.fit(S)
                F = enc.transform(S).toarray().reshape(-1, 1);  # print(2,F.shape, F[0])
                NewX = np.concatenate((NewX, F), axis=1)
                model_meta_data['one-hot_encoded'].append(col)
        else:
            if abs(X[col].skew()) > 0.7:
                X[col] = np.log1p(X[col])
                model_meta_data['log_transformed'].append(col)
            F = X[col].values.reshape(-1, 1);  # print(3,F.shape, F[0])
            NewX = np.concatenate((NewX, F), axis=1)

    # As we need to clean X, add Y and clean to maintain the dimensions
    LE.fit(list(Y))
    NewY = LE.transform(Y)
    NewX = data_to_df(NewX)
    #NewX['Y'] = LE.transform(Y);  # print(list(NewX['Y']))
    #NewX , NewY = clean_dataset(NewX , NewY);  # print(NewX.shape)
    NewX = NewX.iloc[0:, 1:]; # print(NewX.shape)
    #NewY = NewX['Y']
    #NewX = NewX.drop(['Y'], axis=1);  # print(NewX.shape)
    return NewX, NewY, model_meta_data


def compare_models(X,Y):
    model_meta_data = {}
    #prepare data
    NewX, NewY, model_meta_data = prepare_data(X, Y)
    # create validation set
    validation_size = 0.20
    seed = 7
    Xt, Xv, Yt, Yv = model_selection.train_test_split(NewX, NewY,
                test_size=validation_size, random_state=seed)
    #get models
    models = get_models()

    # evaluate each model in turn
    temp = {}
    model_list = {}
    for name, model in models.items():
        kfold = model_selection.KFold(n_splits=2, random_state=seed, shuffle=True)
        cv_results = model_selection.cross_val_score(model, Xt, Yt, cv=kfold, scoring='accuracy')
        cvmean = round(cv_results.mean(), 4)
        cvstd = round(cv_results.std(), 4)
        model_list[name] = {'mean': cvmean, 'std': cvstd}
        temp[name] = cvmean
        # print(name, ':', cvmean, " / ", cvstd); print("")

    final_model = sort_by_value(temp)[0]
    model_meta_data['best_model'] = final_model
    model_meta_data['models_data'] = model_list
    return model_meta_data, NewX



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
        itr = int(data.std())

        bins = []
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
            print("num to cat: ", col);
            x[col] = assert_to_discrete(x[col])
        else:
            print("cat as cat: ", col)
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

