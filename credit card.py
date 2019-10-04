def credit():
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    #Read the dataset from data 
    df = pd.read_csv('C:/Users/Ademola/Desktop/dataset/credit card/crx.data')
    #clean up the messed up data
    df.replace('?',np.NAN,inplace = True)
    #check the null values
    print(df.isnull().sum())
    #fill the numeric null with the mean values
    df.fillna(df.mean(),inplace=True)
    #fill the non numeric values
    df.fillna(method='ffill',inplace=True)
    #check now the null values
    print(df.isnull().sum())
    #convert all string and non numeric codes to numeric encoding
    from sklearn.preprocessing import LabelEncoder
    labelencoder = LabelEncoder()
    for i in df:   
        df[i] = labelencoder.fit_transform(df[i])
    #scale all te dataset to achieve better accuracy
    from sklearn.preprocessing import MinMaxScaler
    sca = MinMaxScaler()
    sca.fit(df)
    sca_features= sca.transform(df)
    df = pd.DataFrame(sca_features,columns=df.columns[:])
        
    df.drop(['00202','f'],axis=1,inplace = True)
    
    X = df.drop('+',axis=1)
    y = df['+']
        
    from sklearn.cross_validation import train_test_split
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=51)
    
    #import your model to use
    from sklearn.linear_model import LogisticRegression
    logmodel = LogisticRegression()
    logmodel.fit(X_train,y_train)
    pred = logmodel.predict(X_test)
    pred = pred.astype(dtype='int64')
    pred2 = labelencoder.inverse_transform(pred)
    #ensure the predicted value is in int form because of the scaling
    y_test = y_test.astype(dtype='int64')
    y_tester = labelencoder.inverse_transform(y_test)
    
    #Display th results in confusion matrix, classification result and accuracy score
    print('The predicted Values', pred2)
    from sklearn.metrics import classification_report
    print('Classification Report: ','\n',classification_report(y_tester,pred2))
    print('\n')
    from sklearn.metrics import confusion_matrix
    print('Confusion Matrix is: ','\n', confusion_matrix(y_tester,pred2))
    print('\n')   
    from sklearn.metrics import accuracy_score
    print('Accuracy Score is:', accuracy_score(y_tester,pred2))
    print('\n')
    
    # Import GridSearchCV
    from sklearn.model_selection import GridSearchCV
    
    # Define the grid of values for tol and max_iter
    TOL = [0.01, 0.001, 0.0001]
    MAX_ITER = [100, 150, 200]
    
    # Create a dictionary
    Param_grid = dict(tol=TOL, max_iter=MAX_ITER)
    
    # Initializing GridSearchCV
    Grid_model = GridSearchCV(estimator=LogisticRegression(), param_grid=Param_grid, cv=5)
    
    # Calculating and summarizing the final results
    Grid_model_result = Grid_model.fit(X, y)
    BEST_SCORE, BEST_PARAMS = Grid_model_result.best_score_, Grid_model_result.best_params_
    print("Best: %f using %s" %  (BEST_SCORE, BEST_PARAMS))
    
    #Try some classifiers
    def other(y_test):
        #decision trees
        from sklearn.tree import DecisionTreeClassifier
        dtree=DecisionTreeClassifier()
        dtree.fit(X_train,y_train)
        pred3 = dtree.predict(X_test)
        pred3 = pred3.astype(dtype='int64')
        pred4 = labelencoder.inverse_transform(pred3)
        #random forest
        from sklearn.ensemble import RandomForestClassifier
        error_rate=[]
        if len(y_tester) <150:
          for i in range(1,len(y_test)):
            rfc = RandomForestClassifier(n_estimators=i)
            rfc.fit(X_train,y_train)
            pred_i = rfc.predict(X_test)
            error_rate.append(np.mean(pred_i!=y_test))
        else:
          for i in range(1,150):
            rfc = RandomForestClassifier(n_estimators=i)
            rfc.fit(X_train,y_train)
            pred_i = rfc.predict(X_test)
            error_rate.append(np.mean(pred_i!=y_test))
        nm = error_rate.index(min(error_rate))
        rfc = RandomForestClassifier(n_estimators=(nm+1))
        rfc.fit(X_train,y_train)
        pred5 = rfc.predict(X_test)
        pred5 = pred5.astype(dtype='int64')
        pred6 = labelencoder.inverse_transform(pred5)
        #trees
        from sklearn.tree import DecisionTreeClassifier
        dtree=DecisionTreeClassifier()
        dtree.fit(X_train,y_train)
        pred7 = dtree.predict(X_test)
        pred7 = pred7.astype(dtype='int64')
        pred8 = labelencoder.inverse_transform(pred7)
        
        #results of the prediction
        from sklearn.metrics import classification_report
        print('Classification Report for trees: ','\n',classification_report(y_test,pred8))
        print('\n')
        from sklearn.metrics import classification_report
        print('Classification Report for decision trees: ','\n',classification_report(y_test,pred4))
        print('\n')
        from sklearn.metrics import classification_report
        print('Classification Report for random forest: ','\n',classification_report(y_test,pred6))
        print('\n')
        from sklearn.metrics import accuracy_score
        print('Accuracy Score for trees is:', accuracy_score(y_test,pred8))
        print('\n')
        from sklearn.metrics import accuracy_score
        print('Accuracy Score for decision trees is:', accuracy_score(y_test,pred4))
        print('\n')
        from sklearn.metrics import accuracy_score
        print('Accuracy Score for random forest is:', accuracy_score(y_test,pred6))
        print('\n')
        #get an input ans for grid search
    ans = input('Do you want to do a grid search for a better model?, it could take a while. Type y/n or yes/no: ')
    if ans == 'y' or ans == 'yes':
        other(y_tester)
    else:
        print('end')
            
if __name__=='__main__':
    credit()               


