
import pandas as pd


def load_data():

    import pandas as pd

    df = pd.read_csv("2023-09-15_duplicate_2.csv")

    df = df[df["last_seen"]>= '2022-01-01']
    df.loc[df['accepted'] > 0.0, 'classification'] = 1
    df.loc[df['accepted'] == 0.0, 'classification'] = 0
    df['classification'] = df['classification'].astype('int32')

    return df



def balance_data():

    df = load_data()

    non_interaction_quantity = len(df[df.classification == 0])
    accepted_quantity = len(df[df.classification == 1])

    non_interaction_data = df[df["classification"] == 0]
    accepted_data = df[df["classification"] == 1]

    non_interaction_factor = 0.9
    accepted_factor = 1 - non_interaction_factor

    non_interaction_quantity = int(accepted_quantity * non_interaction_factor /accepted_factor )



    non_interaction_partion = non_interaction_data.sample(n=non_interaction_quantity)
    balance_df = pd.concat([non_interaction_partion, accepted_data])


    return balance_df


def var_definition():

    df = load_data()
    balance_df = balance_data()

    from sklearn.preprocessing import MinMaxScaler
    from sklearn.feature_selection import SelectKBest, chi2, f_classif
    import matplotlib.pyplot as plt

    y = balance_df["classification"].to_numpy()
    x = balance_df.drop(["classification","last_login","last_seen","validation_state","town","state"], axis=1).to_numpy()

    """Normalizacion de variables entrada"""
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(x)
    x_normal = scaler.transform(x)

    """SelecciÃ³n de variables mÃ¡s relevantes mediante la prueba chi2"""
    selectKBest = SelectKBest(chi2, k=2) # K is var number that choose us
    x = selectKBest.fit_transform(x_normal, y)

    cols = selectKBest.get_support(indices=True)
    variables = balance_df.iloc[:,cols].columns

    x_new = pd.DataFrame(x, columns = variables)
    y_new = pd.DataFrame(y, columns = ["classification"])



    return x_new, y_new, variables



def make_train_test_split(x_new, y_new):

    from sklearn.model_selection import train_test_split

    (x_train, x_test, y_train, y_test) = train_test_split(
                                                          x_new,
                                                          y_new,
                                                          test_size=0.2,
                                                          random_state=123456,
                                                          shuffle = True
                                                      )
    return x_train, x_test, y_train, y_test




def eval_metrics(y_test, y_pred):

    from sklearn.metrics import mean_squared_error,accuracy_score, recall_score, precision_score,balanced_accuracy_score,f1_score,fbeta_score,classification_report
    from sklearn.metrics import auc, roc_auc_score, roc_curve

    asc = accuracy_score(
                          y_true=y_test,
                          y_pred=y_pred,
                          normalize=True,
                          )
    bas = balanced_accuracy_score(y_test,y_pred)
    rs = recall_score(y_test, y_pred)
    ps = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    bs = fbeta_score(y_test, y_pred,average='micro', beta=1),
    cr = classification_report(y_test,y_pred)
    ras = roc_auc_score(y_test,y_pred)


    return bas, asc, rs, ps, f1, bs, cr, ras


def report(estimator,bas, asc, rs, ps, f1, bs, cr, ras):

    # print(estimator, ":", sep="")
    # print("------------------------------------")
    # print(f"El accuracy score es {str(round(asc,4))}")
    # print(f"El accuracy balanced score es {str(round(bas,4))}")
    # print(f"El recall score es {str(round(rs,4))}")
    # print(f"El precision score es {str(round(ps,4))}")
    # print(f"El f1_score es {str(round(f1,4))}")
    # print(f"El fbeta score es {str(round(bs[0],4))}")
    # print(f"El Ã¡rea bajo la curva ROC es {str(round(ras,4))}")
    a=1

def save_best_estimator(estimator):

    import os
    import pickle

    if not os.path.exists("models"):
        os.makedirs("models")
    with open("models/estimator.pickle", "wb") as file:
        pickle.dump(estimator, file)


def load_best_estimator():

    import os
    import pickle

    if not os.path.exists("models"):
        return None
    with open("models/estimator.pickle", "rb") as file:
        estimator = pickle.load(file)

    return estimator


def train_estimator( C, fit_intercept, max_iter,  verbose = 0 ): #Aca van los parametros

    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GridSearchCV


    """Load and split data"""
    x_new, y_new, variables = var_definition()
    x_train, x_test, y_train, y_test = make_train_test_split(x_new, y_new)


    """Search best parameters with cross validation"""

    param_grid = {
                  'C': [0.1, 1, 10],
                  'fit_intercept': [True, False],
                  'max_iter': [100, 500, 1000]
                  }


    clf = LogisticRegression(penalty = 'l2', multi_class='ovr', solver= 'lbfgs', tol = 0.001,random_state = None, verbose = 0, warm_start = False, l1_ratio = None)
    grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=3, verbose=0)


    """ Entrenar """
    grid_search.fit(x_train, y_train.values.ravel())

    """ Evaluar como le fue al modelo """
    estimator = grid_search.best_estimator_
    y_pred = estimator.predict(x_test)

    bas, asc, rs, ps, f1, bs, cr, ras = eval_metrics(y_test, y_pred=estimator.predict(x_test))
    if verbose > 0:
        report(estimator,bas, asc, rs, ps, f1, bs, cr, ras)

    best_estimator = load_best_estimator()
    if best_estimator is None or estimator.score(x_test, y_test) > best_estimator.score(x_test, y_test):
       best_estimator = estimator

    save_best_estimator(best_estimator)

    return grid_search

train_estimator(C=10, fit_intercept=False, max_iter=100,verbose=1)



def check_estimator():

    x_new, y_new, variables = var_definition()
    x_train, x_test, y_train, y_test = make_train_test_split(x_new, y_new)
    estimator = load_best_estimator()
    bas, asc, rs, ps, f1, bs, cr, ras = eval_metrics(y_test, y_pred=estimator.predict(x_test))
    report(estimator, bas, asc, rs, ps, f1, bs, cr, ras)

check_estimator()


def mat_pred():

    from sklearn.linear_model import LogisticRegression

    x_new, y_new, variables = var_definition()
    grid_search = train_estimator(C=10, fit_intercept=False, max_iter=100,verbose=1)


    y_pred_new = grid_search.predict(x_new)
    prediction_df = pd.DataFrame(x_new,columns = variables)
    prediction_df['classification'] = y_new
    prediction_df['pred_classification'] = y_pred_new

    return prediction_df


def prediction_plot():

    import plotly.express as px

    prediction_df = mat_pred()

    classification_counts = prediction_df['classification'].value_counts()
    pred_classification_counts = prediction_df['pred_classification'].value_counts()

    counts_class = pd.DataFrame({'classification':classification_counts, 'pred_classification':pred_classification_counts})
    counts_class = counts_class.reset_index()
    counts_class = counts_class.rename(columns={'index': 'decision'})
    counts_class['decision'] = counts_class['decision'].replace(0,"Not accepted")
    counts_class['decision'] = counts_class['decision'].replace(1,"Accepted")

    fig_predic = px.bar(counts_class, x = 'decision', y=['classification','pred_classification'], barmode = "group", title="Real vs. Prediction swap acept offers")
    fig_predic.show()

    fig_predic.write_image("static/bar_prediction.png", width=800, height=600, scale=2)
