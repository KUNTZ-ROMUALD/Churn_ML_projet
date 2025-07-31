"""
Date :
Author : Kuntz Romuald
"""

import pandas as pd
import numpy as np
import logging 
import seaborn as sns 
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report,RocCurveDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import FunctionTransformer,StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer 


import joblib


#pour la journalisation 
logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s-%(levelname)s-%(message)s'
)


#importation des données 
def import_data(path):
    """
    Retourne un Dataframe à partir d'un chemin de fichier csv 
    
    input : 
            path
    output : 
            dataFrame 

    """
    df=pd.read_csv(path)
    df.drop(columns='customerID',axis=1,inplace=True)
    df['Churn']=df['Churn'].apply(lambda row: 0 if row=="No" else 1 )


    return df 

def data_spliting(df):

    """
    Retourne un à partir d'un dataFrame les données de test,validation et d'entrainement

    input : 
            dataFrame 

    output :
            train data 
            validate data
            test data 
    """
    #division des données 
    train, test = train_test_split(df,test_size=0.3,random_state=111,stratify=df['Churn'])
    test,validate= train_test_split(test,test_size=0.5,stratify=test['Churn'],random_state=111)

    #Enregistrement dans les fichiers 
    train.to_csv('./data/train.csv',index=False)
    test.to_csv('./data/test.csv',index=False)
    validate.to_csv('./data/validate.csv',index=False)

    x_train=train.drop(columns='Churn',axis=1)
    y_train=train['Churn']
    x_val= validate.drop(columns='Churn',axis=1)
    y_val=validate['Churn']


    return train,x_train,y_train,x_val,y_val

#Fonction pour convertir la variable TotalCharges 
def convert_TotalCharges(data):
    data_copy=data.copy()
    data_copy['TotalCharges']=pd.to_numeric(data_copy['TotalCharges'],errors='coerce')
    return data_copy.values

def perfom_eda(df):
    """
    input : 
            dataFrame 
    output :
            None

    """

    df_copy=df.copy()
    list_columns=df_copy.columns.to_list()
    list_columns.append('Heatmap')
    df_corr=df_copy.corr(numeric_only=True)
    for column_name in list_columns :
        plt.figure(figsize=(10,6))
        if column_name == "Heatmap" :
            sns.heatmap(
                df_corr,
                mask=np.triu(np.ones_like(df_corr,dtype=bool)),
                center=0,cmap='RdBu',linewidths=1,annot=True,
                fmt='.2f',vmin=-1,vmax=1
            )
        else :
            if df[column_name].dtype != 'o' :
                df[column_name].hist()
            else : 
                sns.countplot(data=df,x=column_name)
        plt.savefig("images/eda/"+column_name+".jpg")
        plt.close()
 

def classication_repport_images(y_train,y_train_preds,y_val,y_val_preds):

    '''
    Produire un rapport de classication pour les resultats d'entrainement et de validation 
    et fait une sauvegarde des rapports 

    input :
            y_train 
            y_train_preds
            x_val
            y_val_preds 
    output : 
            None 
    
    '''
    class_repport_disco= {
        'Logistic Regression train results' : classification_report(y_train,y_train_preds),
        'Logistic regression validate results' : classification_report(y_val,y_val_preds)
    }

    for tittle,report in class_repport_disco.items():
        plt.rc('figure',figsize=(7,3))
        plt.text(0.2,0.3,str(report),
                 {'fontsize':10},fontproperties='monsapce')
        plt.axis('off')
        plt.title(tittle,fontweight='bold')
        plt.savefig('images/results/'+tittle+'.jpg')
        plt.close()

#Création du piple de modélisation 

def build_pipeline ():
    """
    Build a pipeline 

    """

    categorical_features=['gender',
    'Partner',
    'Dependents',
    'PhoneService',
    'MultipleLines',
    'InternetService',
    'OnlineSecurity',
    'OnlineBackup',
    'DeviceProtection',
    'TechSupport',
    'StreamingTV',
    'StreamingMovies',
    'Contract',
    'PaperlessBilling',
    'PaymentMethod']

    numerical_features=['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges']

    #Pipeline de prétraitement des variables indépendantes numériques 
    numeric_transformer=Pipeline(
    steps=[('convert',FunctionTransformer(convert_TotalCharges)),
           ('imputer',SimpleImputer(strategy='median')),
           ('scaler',StandardScaler())]
    )
    #Pipeline de prétraitement des variables  catégorielle 
    categorical_transformer=Pipeline(
    steps=[('oneHotEncoder',OneHotEncoder(sparse_output=False,handle_unknown='ignore'))]
    )

    #Concatenation des deux piples
    preprocessor=ColumnTransformer(
    transformers=[('numeric',numeric_transformer,numerical_features),
                  ('categorical',categorical_transformer,categorical_features)]
    )  

    #Pipeline de modélisation 
    pipeline_model=Pipeline(
        steps=[('preprocessor',preprocessor),
               ('logreg',LogisticRegression(random_state=111,max_iter=2000,C=0.5,solver='lbfgs'))]
    )
    return pipeline_model


def train_models(x_train,x_val,y_train,y_val):
    """
    entraine et charge le resultat 

    input :
            x_train
            y_train
            x_val
            y_val 
    output :
            None 
    """
    #Formation du modèle 
    model=build_pipeline()
    model.fit(x_train,y_train)

    #Prédiction
    y_tain_preds=model.predict(x_train)
    y_val_preds=model.predict(x_val)

    #ROC courbe images 

    lrc_plot=RocCurveDisplay.from_estimator(model,x_val,y_val)
    plt.savefig("images/results/roc_image.jpg")
    plt.close()

    #Classification repport images 
    classication_repport_images(y_train,y_tain_preds,y_val,y_val_preds)

    #Sauvegarder le model
    joblib.dump(model,'./models/logreg_model.pk1')


def main():
    logging.info("Importation des données")
    raw_data=import_data("./data/data_test_telecom.csv")
    logging.info("Impotation des données : SUCCESS")

    logging.info("Division des données ...")
    train_data,xtrain,ytrain,xval,yval=data_spliting(raw_data)
    logging.info('Division des données : SUCCES')
   
    logging.info("Analyse exploiratoire")
    perfom_eda(train_data)
    logging.info('Analyse exploiratoire des données : SUCCESS')

    logging.info('Formation du model')
    train_models(xtrain,xval,ytrain,yval)
    logging.info('Formation du model : SUCCESS')

if __name__== "__main__" :
    print("Exécution en cours")
    main()
    print("Fin de l'exécution")
