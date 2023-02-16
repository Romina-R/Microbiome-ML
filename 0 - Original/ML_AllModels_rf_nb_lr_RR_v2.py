#!/usr/bin/env python
# coding: utf-8

# In[1]:

def Run_ML_Models(have_dog,sample_type,human_role):
    import pandas as pd
    import numpy as np
    import os
    import matplotlib.pyplot as plt

    from sklearn import tree
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import validation_curve

    from sklearn.model_selection import train_test_split
    from sklearn.metrics import confusion_matrix

    from sklearn.naive_bayes import GaussianNB
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import accuracy_score


    # In[2]:


    mergedf = pd.read_csv('db/DB_CSV_merged_v1.csv', low_memory=False)

    mergedf = mergedf.drop("Unnamed: 0", axis=1)
    mergedf.shape


    # # Filtering Train and Test Datasets

    # In[3]:


    #FILTER 1
    # have_dog = "only_non_dog_owners"

    #FILTER 2
    # sample_type = "skin"
    # sample_type = "all"
    # #FILTER 3
    # # not_sample_type = "stool"

    # # #FILTER 4
    # human_role = "Partner"
    # human_role = "all"
    # #FILTER 5
    # not_human_role = "Offspring"

    #FILTER 6
    # not_familyID = 63


    # # SPLITTING TRAIN and TEST (test = dogs, train = humans)

    # In[4]:


    human_data_train = mergedf[mergedf.host_common_name =="human"]

    dog_data_test = mergedf[mergedf.host_common_name =="dog"]
    dog_data_test.head(1)


    # In[5]:


    # Applying filters above
    try:
        if have_dog == "only_dog_owners":
            human_data_train = human_data_train[human_data_train.have_dog =="yes"]
        
        elif have_dog == "only_non_dog_owners":
            human_data_train = human_data_train[human_data_train.have_dog =="no"]
        elif have_dog == "all":
            human_data_train = human_data_train
        else:
            human_data_train = human_data_train
    except:
        print("no filter here 1")
    print(human_data_train.shape)

    try:    
        if sample_type!= "all":
            human_data_train = human_data_train[human_data_train.sample_type ==sample_type]
            dog_data_test = dog_data_test[dog_data_test.sample_type ==sample_type]
        elif sample_type == "all":
            human_data_train = human_data_train
            dog_data_test = dog_data_test
        else:
            human_data_train = human_data_train
            dog_data_test = dog_data_test
    except:
        print("no filter here 2")
    print(human_data_train.shape)  

    # try:    
    #     if not_sample_type:
    #         human_data_train = human_data_train[human_data_train.sample_type !=not_sample_type]
    #         dog_data_test = dog_data_test[dog_data_test.sample_type !=not_sample_type]
    #     elif sample_type == "all":
    #         human_data_train = human_data_train
    #         dog_data_test = dog_data_test
    # except:
    #     print("no filter here 3")
        
    try:    
        if human_role != "all":
            human_data_train = human_data_train[human_data_train.role ==human_role]
        elif human_role == "all":
            human_data_train = human_data_train
        else:
            human_data_train = human_data_train
    except:
        print("no filter here 4")
    print(human_data_train.shape)    


    # # Train/test split

    # In[6]:


    X_train = human_data_train.iloc[:, -1035:-1]
    y_train = human_data_train["familyID"].values.reshape(-1, 1)

    y_test = dog_data_test["familyID"].values.reshape(-1, 1)
    X_test = dog_data_test.iloc[:, -1035:-1]


    # In[7]:


    # X = human_data_train.iloc[:, -1035:-1]
    # y = human_data_train["familyID"].values.reshape(-1, 1)

    X_train_h, X_test_h, y_train_h, y_test_h = train_test_split(X_train, y_train, random_state=42)

    # train_h, test_h = train_test_split(human_data_train, random_state=42)

    # X_train_h = X_train_h.iloc[:, -1035:-1]
    # X_test_h = X_test_h.iloc[:, -1035:-1]

    # y_train_h = train_h.iloc["familyID"].values.reshape(-1, 1)
    # y_test_h = test_h.iloc["familyID"].values.reshape(-1, 1)


    # # Hyperparameter Selection

    # In[8]:


    n_estimators = 1000

    from sklearn.ensemble import RandomForestClassifier
    rf = RandomForestClassifier(n_estimators=n_estimators)

    #for Humans
    rf_h = RandomForestClassifier(n_estimators=n_estimators)


    # # Non-Scaled Model Fitting

    # In[9]:


    # DOG --> HUMAN
    #from non-scaled data
    rf = rf.fit(X_train, y_train.ravel())
    rf.score(X_test, y_test) #geni impurity coefficient

    dog_rf = round(rf.score(X_test, y_test),2)

    print(f'Test Score: {dog_rf}')
    print(f'Train Score: {rf.score(X_train, y_train)}')


    # In[10]:


    # Human --> Human
    rf_h = rf_h.fit(X_train_h, y_train_h.ravel())
    rf_h.score(X_test_h, y_test_h) #geni impurity coefficient

    hu_rf = round(rf_h.score(X_test_h, y_test_h),2)

    print(f'Test Score for Human (RF): {hu_rf}')
    print(f'Train Score: {rf_h.score(X_train_h, y_train_h)}')


    # # Scaled X Model Fitting

    # In[11]:


    # from sklearn.preprocessing import StandardScaler
    # X_scaler = StandardScaler().fit(X_train)

    # X_train_scaled = X_scaler.transform(X_train)
    # X_test_scaled = X_scaler.transform(X_test)


    # In[12]:


    # # from scaled data
    # rf_scaled = RandomForestClassifier(n_estimators=n_estimators)
    # rf_scaled = rf_scaled.fit(X_train_scaled, y_train.ravel())
    # rf_scaled.score(X_test_scaled, y_test) #geni impurity coefficient

    # print(f'Test Score: {rf_scaled.score(X_test_scaled, y_test)}')
    # print(f'Train Score: {rf_scaled.score(X_train_scaled, y_train)}')


    # # Evaluate most-guessed family for each dog (7 collection site/predictions per animal) ==> 1 prediction

    # In[13]:


    y_pred = rf.predict(X_test)

    outputdf = pd.DataFrame({"SampleName": dog_data_test["sample_name"],"Anonymized_Name":dog_data_test["anonymized_name"],"Prediction": y_pred, "Actual": y_test.ravel()}).reset_index(drop=True)
    print(outputdf.shape)
    outputdf.head()


    # In[14]:


    sumOutput = outputdf.groupby(["Anonymized_Name","Actual","Prediction"]).count()
    sumOutput.reset_index()
    sumOutput.head()


    # In[15]:


    # sumOutput.groupby(["Anonymized_Name"])['SampleName'].max()
    idx = sumOutput.groupby(["Anonymized_Name"])['SampleName'].transform(max) == sumOutput['SampleName']
    maxOutput = sumOutput[idx]
    maxOutput = maxOutput.reset_index()
    print(maxOutput.shape)
    # maxOutput

    maxOutput_noDup = maxOutput.drop_duplicates(subset=['Anonymized_Name'], keep="first")
    print(maxOutput_noDup.shape)
    # maxOutput_noDup

    maxOutput_C = maxOutput_noDup[maxOutput_noDup.Actual == maxOutput_noDup.Prediction]

    print(maxOutput_C.shape)
    # maxOutput_C

    maxOutput_I = maxOutput_noDup[maxOutput_noDup.Actual != maxOutput_noDup.Prediction]

    print(maxOutput_I.shape)
    # maxOutput_I


    # In[ ]:





    # In[16]:


    correct_total = maxOutput_C["Anonymized_Name"].count()
    incorrect_total = maxOutput_I["Anonymized_Name"].count()

    accuracy_maxed_pred_rf = correct_total/(correct_total + incorrect_total)
    print(f' Accuracy of most-predicted (dog) family: {round(accuracy_maxed_pred_rf,3)}')


    # #NOW FOR HUMANS

    # # LOGISTIC REGRESSION

    # In[17]:


    from sklearn.preprocessing import StandardScaler
    X_scaler = StandardScaler().fit(X_train)

    X_train_scaled = X_scaler.transform(X_train)
    X_test_scaled = X_scaler.transform(X_test)

    X_train_scaled_h = X_scaler.transform(X_train_h)
    X_test_scaled_h = X_scaler.transform(X_test_h)


    # In[18]:


    from sklearn.linear_model import LogisticRegression
    classifier = LogisticRegression()
    classifier


    # In[19]:


    ############ dog logistic regression
    model = classifier.fit(X_train_scaled, y_train.ravel())

    dog_lr = round(classifier.score(X_test_scaled, y_test),2)
    print(f'Test Data Model Score for Dogs: {dog_lr}')

    ############# human logistic regression
    model_h = classifier.fit(X_train_scaled_h, y_train_h.ravel())

    hu_lr = round(classifier.score(X_test_scaled_h, y_test_h),2)
    print(f'Test Data Model Score for Humans: {hu_lr}')


    # In[20]:


    y_pred_logisticR = model.predict(X_test_scaled)

    # y_pred_logisticR_h = model_h.predict(X_test_scaled_h)

    outputdf = pd.DataFrame({"SampleName": dog_data_test["sample_name"],"Anonymized_Name":dog_data_test["anonymized_name"],"Prediction": y_pred_logisticR, "Actual": y_test.ravel()}).reset_index(drop=True)
    print(outputdf.shape)
    outputdf.head()


    # In[21]:


    # feature_importance=pd.DataFrame(np.hstack(np.array(bacteria_array, coef), columns=['feature_bacteria', 'importance'])


    # In[22]:


    # # feature_importance=pd.DataFrame(list(zip(bacteria_array.T, model.coef_.T)), columns=['feature', 'importance'])
    # feature_importance=pd.DataFrame(list(zip(bacteria_array, coef)), columns=['feature', 'importance'])
    # feature_importance


    # In[23]:


    sumOutput_logisticR = outputdf.groupby(["Anonymized_Name","Actual","Prediction"]).count()
    sumOutput_logisticR.reset_index()
    sumOutput_logisticR.head()


    # In[24]:


    # sumOutput.groupby(["Anonymized_Name"])['SampleName'].max()
    idx_logisticR = sumOutput_logisticR.groupby(["Anonymized_Name"])['SampleName'].transform(max) == sumOutput_logisticR['SampleName']
    maxOutput_logisticR = sumOutput_logisticR[idx_logisticR]
    maxOutput_logisticR = maxOutput_logisticR.reset_index()
    print(maxOutput_logisticR.shape)
    # maxOutput


    # In[25]:


    maxOutput_noDup_logisticR = maxOutput_logisticR.drop_duplicates(subset=['Anonymized_Name'], keep="first")
    print(maxOutput_noDup_logisticR.shape)
    # maxOutput_noDup

    maxOutput_C_logisticR = maxOutput_noDup_logisticR[maxOutput_noDup_logisticR.Actual == maxOutput_noDup_logisticR.Prediction]

    print(maxOutput_C_logisticR.shape)
    # maxOutput_C

    maxOutput_I_logisticR = maxOutput_noDup_logisticR[maxOutput_noDup_logisticR.Actual != maxOutput_noDup_logisticR.Prediction]

    print(maxOutput_I_logisticR.shape)
    # maxOutput_I


    # In[26]:


    correct_total_logisticR = maxOutput_C_logisticR["Anonymized_Name"].count()
    incorrect_total_logisticR = maxOutput_I_logisticR["Anonymized_Name"].count()

    accuracy_maxed_pred_logisticR_dog = correct_total_logisticR/(correct_total_logisticR + incorrect_total_logisticR)
    # print(f' Accuracy of most-predicted (dog) family (logistic Regression): {round(accuracy_maxed_pred_logisticR,2)}')


    # # Naive Bayes

    # In[27]:


    model_nb_dog = GaussianNB()
    model_nb_dog.fit(X_train, y_train.ravel())


    # In[28]:


    pred_nb_dog = model_nb_dog.predict(X_test)
    accuracy_nb_dog = round(accuracy_score(y_test, pred_nb_dog),2)


    # print(f' Accuracy of most-predicted (dog) family (NB): {accuracy_nb_dog}')


    # In[29]:


    model_nb_hu = GaussianNB()
    model_nb_hu.fit(X_train_h, y_train_h.ravel())


    # In[30]:


    pred_nb_hu = model_nb_hu.predict(X_test_h)
    accuracy_nb_hu = round(accuracy_score(y_test_h, pred_nb_hu),2)


    # In[31]:


    print(f'Test Score for Human (RF): {hu_rf}')
    print(f' Accuracy of most-predicted (dog) family: {round(accuracy_maxed_pred_rf,3)}')

    print(f'Test Data Model Score for Humans (logistic Regression): {hu_lr}')
    print(f' Accuracy of most-predicted (dog) family (logistic Regression): {round(accuracy_maxed_pred_logisticR_dog,2)}')

    print(f'Test Data Model Score for Humans (Naive Bayes): {accuracy_nb_hu}')
    print(f' Accuracy of most-predicted (dog) family (Naive Bayes): {accuracy_nb_dog}')
    accuracy_maxed_pred_rf = 100*(round(accuracy_maxed_pred_rf,2))

    resultsDict = {'randomForest': f'{accuracy_maxed_pred_rf}%' , 'logisticRegression':f'{100*(round(accuracy_maxed_pred_logisticR_dog,2))}%', 'naiveBayes_dog':f'{100*(round(accuracy_nb_dog,2))}%', 'randomForest_human':f'{round(100*(hu_rf),2)}%', 'logisticRegression_human': f'{100*(hu_lr)}%', 'naiveBayes_human':f'{round(100*(accuracy_nb_hu),2)}%'}
    # resultsDict = {'randomForest_dog': round(accuracy_maxed_pred_rf,2) , 'logisticRegression_dog':round(accuracy_maxed_pred_logisticR,2), 'naiveBayes_dog':accuracy_nb_dog, 'randomForest_human':hu_rf, 'logisticRegression_human': hu_lr, 'naiveBayes_human':accuracy_nb_hu}

    return resultsDict
# In[ ]:




