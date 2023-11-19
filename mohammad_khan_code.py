'''
NAME: Mohammad Saifullah Khan
ROLl NO.: 21169
EECS
ECS308
'''



import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier

from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, RobustScaler, OneHotEncoder
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report


class classification():
    def __init__(self, data_path, label_path, test_data_path, classifier='lsvc'):
        self.data_path = data_path
        self.label_path = label_path
        self.test_data_path = test_data_path
        self.classifier = classifier
        
    def classification_pipeline(self):
        #Random Forest Classifier
        if self.classifier == 'rf':
            print('\n\t Training Random Forest Classifier  \n')
            classifier = RandomForestClassifier(random_state=42)
            classifier_param = {
                'classifier__n_estimators': (20, 50, 100),
                'classifier__criterion': ('gini', 'entropy'),
                'classifier__max_features': ('auto', 'sqrt', 'log2'),
                'classifier__max_depth': (10, 40, 45, 60, 300),
            }

        #AdaBoost Classifier
        if self.classifier == 'ada':
            print('\n\t Training AdaBoost Classifier \n')
            e1 = LogisticRegression(solver='liblinear',class_weight='balanced') 
            e2 = DecisionTreeClassifier(max_depth=50)
            classifier = AdaBoostClassifier(random_state=42)
            classifier_param = {
                'classifier__base_estimator':(e1, e2),
                'classifier__n_estimators': (20, 50, 100),
                'classifier__algorithm': ('SAMME', 'SAMME.R'),
                'classifier__learning_rate': (0.1, 0.5, 1.0),
            }

        #MLP Classifier
        if self.classifier == 'mlp':
            print('\n\t Training MLP Classifier \n')
            classifier = MLPClassifier(random_state=42)
            classifier_param = {
                'classifier__hidden_layer_sizes': ((32, 64, 32), (50, 100, 50), (100,)),
                'classifier__activation': ('logistic', 'relu'),
                'classifier__solver': ('sgd', 'adam'),
                'classifier__alpha': (0.0001, 0.05),
                'classifier__learning_rate': ('invscaling', 'adaptive'),
            }

        #KNN Classifier
        if self.classifier == 'knn':
            print('\n\t Training KNN Classifier \n')
            classifier = KNeighborsClassifier()
            classifier_param = {
                'classifier__n_neighbors': (3, 5, 7, 9),
                'classifier__weights': ('uniform', 'distance'),
                'classifier__algorithm': ('auto', 'ball_tree', 'kd_tree', 'brute'),
            }
        
        #Logistic Regression Classifier
        if self.classifier == 'lr':
            print('\n\t Training Logistic Regression Classifier \n')
            classifier = LogisticRegression(random_state=42)
            classifier_param = {
                'classifier__penalty': ('l1', 'l2'),
                'classifier__C': (0.01, 0.1, 1, 10),
                'classifier__solver': ('liblinear', 'saga'),
            }
        
        #SVM Classifier
        if self.classifier == 'svm':
            print('\n\t Training SVM Classifier \n')
            classifier = SVC(random_state=42)
            classifier_param = {
                'classifier__C': (0.1, 1, 10),
                'classifier__kernel': ('linear', 'poly', 'rbf', 'sigmoid'),
                'classifier__degree': (2, 3, 4),
            }
        
        #Decision Tree Classifier
        if self.classifier == 'dt':
            print('\n\t Training Decision Tree Classifier \n')
            classifier = DecisionTreeClassifier(random_state=42)
            classifier_param = {
                'classifier__criterion': ('gini', 'entropy'),
                'classifier__max_features': ('auto', 'sqrt', 'log2'),
                'classifier__max_depth': (10, 40, 45, 60, 300),
            }
        
        #Linear SVC Classifier
        if self.classifier == 'lsvc':
            print('\n\t Training Linear SVC Classifier \n')
            classifier = LinearSVC(random_state=42)
            classifier_param = {
                'classifier__penalty': ('l1', 'l2'),
                'classifier__C': (0.01, 0.1, 1, 10),
            }
        return classifier, classifier_param

    #Loads the data
    def data(self):
        print('\n\t Loading Data \n')
        data = pd.read_csv(self.data_path)
        labels = pd.read_csv(self.label_path, header=None)
        return data, labels
    

    def classification(self):
        #Loading the data
        data, labels = self.data()

        #Finding the categorical and numerical columns
        threshold_cat_value = 2
        cat_col = [col for col in data.columns if data[col].nunique() <= threshold_cat_value]
        tot_col = [col for col in data.columns if data[col].dtype in ['int64', 'float64']]
        num_col = [col for col in tot_col if col not in cat_col]

        # rbs = RobustScaler()
        # stds = StandardScaler()
        # ohe = OneHotEncoder()
        
        #imputation
        imp = SimpleImputer(strategy='mean')
        data[num_col] = imp.fit_transform(data[num_col])
        
        best =[] 

        #Creating the pipeline
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        for train_index, test_index in skf.split(data, labels):
            X_train, X_test = data.iloc[train_index], data.iloc[test_index]
            y_train, y_test = labels.iloc[train_index], labels.iloc[test_index]

            classifier_obj, classifier_param = self.classification_pipeline()

        #creating preprocessing steps
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', RobustScaler(), num_col),
                    ('cat', OneHotEncoder(), cat_col)
                ]
            )

        #Creating the pipeline
            pipe = Pipeline([
                ('preprocessor', preprocessor),
                ('feature_selection', SelectKBest(f_classif, k=39)),
                ('smote', SMOTE(random_state=42)),
                ('classifier', classifier_obj)
            ])

        #Creating the classifier and classifier parameters
            grid = GridSearchCV(pipe, classifier_param, scoring='f1_macro', cv=5)
            grid.fit(X_train, y_train.values.ravel())
            classifier = grid.best_estimator_
            predicted = classifier.predict(X_test)

            print('\n\n Classification Report \n')
            print(classification_report(y_test, predicted))

            print('\n\n Confusion Matrix \n')
            print(confusion_matrix(y_test, predicted))

            print('\n\n Accuracy Score \n')
            print(accuracy_score(y_test, predicted))

            print('\n\n Precision Score \n')
            print(precision_score(y_test, predicted, average='macro'))

            print('\n\n Recall Score \n')
            print(recall_score(y_test, predicted, average='macro'))

            print('\n\n F1 Score \n')
            print(f1_score(y_test, predicted, average='macro'))

            print('\n\n Parameters \n')
            print(grid.best_params_)

            print('\n\n Estimator \n')
            print(grid.best_estimator_)

            print('\n\n Score \n')
            print(grid.best_score_)
            print('---------*************---------')

            #write results and best params to a file
            print('\n\n Writing results to file \n')
            with open('results_lsvc.txt', 'a') as f:
                f.write('\n\n Classification Report \n')
                f.write(classification_report(y_test, predicted))

                f.write('\n\n Confusion Matrix \n')
                f.write(str(confusion_matrix(y_test, predicted)))

                f.write('\n\n Accuracy Score \n')
                f.write(str(accuracy_score(y_test, predicted)))

                f.write('\n\n Precision Score \n')
                f.write(str(precision_score(y_test, predicted, average='macro')))

                f.write('\n\n Recall Score \n')
                f.write(str(recall_score(y_test, predicted, average='macro')))

                f.write('\n\n F1 Score \n')
                f.write(str(f1_score(y_test, predicted, average='macro')))

                f.write('\n\n Parameters \n')
                f.write(str(grid.best_params_))
                
                f.write('\n\n Estimator \n')
                f.write(str(grid.best_estimator_))
                
                f.write('\n\n Score \n')
                f.write(str(grid.best_score_))
                f.write('\n---------*************---------\n')
        f.close()
        print('\n\n Done \n')















    

 
        
