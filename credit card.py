def credit():
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np

    df = pd.read_csv('C:/Users/Ademola/Desktop/dataset/credit card/crx.data')
    df1 = pd.read_csv('C:/Users/Ademola/Desktop/dataset/credit card/cc_data.csv')

    df.replace('?',np.NAN,inplace = True)
    
    print(df.isnull().sum())
    
    df.fillna(df.mean(),inplace=True)
    
    df.fillna(method='ffill',inplace=True)
    
    print(df.isnull().sum())
    
    
    from sklearn.preprocessing import LabelEncoder
    labelencoder = LabelEncoder()
    for i in df:   
        df[i] = labelencoder.fit_transform(df[i])
        
    dff = df.copy()
    df.drop(['00202','f'],axis=1,inplace = True)
    
    X = df.drop('+',axis=1)
    y = df['+']
    
    from sklearn.cross_validation import train_test_split
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=51)
    
    from sklearn.linear_model import LogisticRegression
    logmodel = LogisticRegression()
    logmodel.fit(X_train,y_train)
    pred = logmodel.predict(X_test)
    pred2 = labelencoder.inverse_transform(pred)
    
    y_tester = labelencoder.inverse_transform(y_test)
    yy = labelencoder.inverse_transform(y)
    
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
    #Try some classifiers
    def grid(y_test):
        #decision trees
        from sklearn.tree import DecisionTreeClassifier
        dtree=DecisionTreeClassifier()
        dtree.fit(X_train,y_train)
        pred3 = dtree.predict(X_test)
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
        pred6 = labelencoder.inverse_transform(pred5)
        #trees
        from sklearn.tree import DecisionTreeClassifier
        dtree=DecisionTreeClassifier()
        dtree.fit(X_train,y_train)
        pred7 = dtree.predict(X_test)
        pred8 = labelencoder.inverse_transform(pred7)
        
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
    ans = input('Do you want to do a grid search for a better model?,could take a while. Type y/n or yes/no: ')
    if ans == 'y' or ans == 'yes':
        grid(y_tester)
    else:
        print('end')
            
if __name__=='__main__':
    credit()               




