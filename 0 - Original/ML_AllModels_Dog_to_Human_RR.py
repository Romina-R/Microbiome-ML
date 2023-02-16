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


    # In[2]:


    mergedf = pd.read_csv('db/DB_CSV_merged_v1.csv', low_memory=False)

    mergedf = mergedf.drop("Unnamed: 0", axis=1)
    mergedf.head(1)


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
    # # human_role = "Partner"
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


    # In[6]:


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

    # In[7]:


    X_train = human_data_train.iloc[:, -1035:-1]
    y_train = human_data_train["familyID"].values.reshape(-1, 1)

    y_test = dog_data_test["familyID"].values.reshape(-1, 1)
    X_test = dog_data_test.iloc[:, -1035:-1]


    # # Hyperparameter Selection

    # In[8]:


    n_estimators = 1000

    from sklearn.ensemble import RandomForestClassifier
    rf = RandomForestClassifier(n_estimators=n_estimators)


    # # Non-Scaled Model Fitting

    # In[9]:


    #from non-scaled data
    rf = rf.fit(X_train, y_train.ravel())
    rf.score(X_test, y_test) #geni impurity coefficient

    print(f'Test Score: {rf.score(X_test, y_test)}')
    print(f'Train Score: {rf.score(X_train, y_train)}')


    # # Scaled X Model Fitting

    # In[10]:


    # from sklearn.preprocessing import StandardScaler
    # X_scaler = StandardScaler().fit(X_train)

    # X_train_scaled = X_scaler.transform(X_train)
    # X_test_scaled = X_scaler.transform(X_test)


    # In[11]:


    # # from scaled data
    # rf_scaled = RandomForestClassifier(n_estimators=n_estimators)
    # rf_scaled = rf_scaled.fit(X_train_scaled, y_train.ravel())
    # rf_scaled.score(X_test_scaled, y_test) #geni impurity coefficient

    # print(f'Test Score: {rf_scaled.score(X_test_scaled, y_test)}')
    # print(f'Train Score: {rf_scaled.score(X_train_scaled, y_train)}')


    # # Evaluate most-guessed family for each dog (7 collection site/predictions per animal) ==> 1 prediction

    # In[12]:


    y_pred = rf.predict(X_test)

    outputdf = pd.DataFrame({"SampleName": dog_data_test["sample_name"],"Anonymized_Name":dog_data_test["anonymized_name"],"Prediction": y_pred, "Actual": y_test.ravel()}).reset_index(drop=True)
    print(outputdf.shape)
    outputdf.head()


    # In[13]:


    sumOutput = outputdf.groupby(["Anonymized_Name","Actual","Prediction"]).count()
    sumOutput.reset_index()
    sumOutput.head()


    # In[14]:


    # sumOutput.groupby(["Anonymized_Name"])['SampleName'].max()
    idx = sumOutput.groupby(["Anonymized_Name"])['SampleName'].transform(max) == sumOutput['SampleName']
    maxOutput = sumOutput[idx]
    maxOutput = maxOutput.reset_index()
    print(maxOutput.shape)
    # maxOutput


    # In[15]:


    maxOutput_noDup = maxOutput.drop_duplicates(subset=['Anonymized_Name'], keep="first")
    print(maxOutput_noDup.shape)
    # maxOutput_noDup


    # In[16]:


    maxOutput_C = maxOutput_noDup[maxOutput_noDup.Actual == maxOutput_noDup.Prediction]

    print(maxOutput_C.shape)
    # maxOutput_C


    # In[17]:


    maxOutput_I = maxOutput_noDup[maxOutput_noDup.Actual != maxOutput_noDup.Prediction]

    print(maxOutput_I.shape)
    # maxOutput_I


    # In[18]:


    correct_total = maxOutput_C["Anonymized_Name"].count()
    incorrect_total = maxOutput_I["Anonymized_Name"].count()

    accuracy_maxed_pred_rf = correct_total/(correct_total + incorrect_total)
    print(f' Accuracy of most-predicted (dog) family: {round(accuracy_maxed_pred_rf,3)}')


    # # LOGISTIC REGRESSION

    # In[19]:


    from sklearn.preprocessing import StandardScaler
    X_scaler = StandardScaler().fit(X_train)

    X_train_scaled = X_scaler.transform(X_train)
    X_test_scaled = X_scaler.transform(X_test)


    # In[20]:


    from sklearn.linear_model import LogisticRegression
    classifier = LogisticRegression()
    classifier


    # In[21]:


    model = classifier.fit(X_train_scaled, y_train.ravel())
    print(f'Test Data Model Score: {round(classifier.score(X_test_scaled, y_test),2)}')


    # In[22]:


    y_pred_logisticR = model.predict(X_test_scaled)

    outputdf = pd.DataFrame({"SampleName": dog_data_test["sample_name"],"Anonymized_Name":dog_data_test["anonymized_name"],"Prediction": y_pred_logisticR, "Actual": y_test.ravel()}).reset_index(drop=True)
    print(outputdf.shape)
    outputdf.head()


    # In[23]:


    sumOutput_logisticR = outputdf.groupby(["Anonymized_Name","Actual","Prediction"]).count()
    sumOutput_logisticR.reset_index()
    sumOutput_logisticR.head()


    # In[25]:


    # sumOutput.groupby(["Anonymized_Name"])['SampleName'].max()
    idx_logisticR = sumOutput_logisticR.groupby(["Anonymized_Name"])['SampleName'].transform(max) == sumOutput_logisticR['SampleName']
    maxOutput_logisticR = sumOutput_logisticR[idx_logisticR]
    maxOutput_logisticR = maxOutput_logisticR.reset_index()
    print(maxOutput_logisticR.shape)
    # maxOutput


    # In[26]:


    maxOutput_noDup_logisticR = maxOutput_logisticR.drop_duplicates(subset=['Anonymized_Name'], keep="first")
    print(maxOutput_noDup_logisticR.shape)
    # maxOutput_noDup

    maxOutput_C_logisticR = maxOutput_noDup_logisticR[maxOutput_noDup_logisticR.Actual == maxOutput_noDup_logisticR.Prediction]

    print(maxOutput_C_logisticR.shape)
    # maxOutput_C

    maxOutput_I_logisticR = maxOutput_noDup_logisticR[maxOutput_noDup_logisticR.Actual != maxOutput_noDup_logisticR.Prediction]

    print(maxOutput_I_logisticR.shape)
    # maxOutput_I


    # In[27]:


    correct_total_logisticR = maxOutput_C_logisticR["Anonymized_Name"].count()
    incorrect_total_logisticR = maxOutput_I_logisticR["Anonymized_Name"].count()

    accuracy_maxed_pred_logisticR = correct_total_logisticR/(correct_total_logisticR + incorrect_total_logisticR)
    print(f' Accuracy of most-predicted (dog) family: {round(accuracy_maxed_pred_logisticR,3)}')


    # In[ ]:

    resultsDict = {'randomForest': round(accuracy_maxed_pred_rf,3) , 'logisticRegression':round(accuracy_maxed_pred_logisticR,3)}

    return resultsDict
    # return {'randomForest': round(accuracy_maxed_pred_rf,3) , 'logisticRegression':round(accuracy_maxed_pred_logisticR,3)}



