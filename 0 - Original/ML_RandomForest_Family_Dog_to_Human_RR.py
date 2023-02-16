#!/usr/bin/env python
# coding: utf-8

# In[1]:


def RandomForestRR(have_dog,sample_type,human_role):
#   return jsonify(round(accuracy_maxed_pred,3))

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


    # In[3]:


    mergedf.body_site.value_counts()


    # # Filtering Train and Test Datasets

    # In[4]:


    #FILTER 1
    # have_dog = "all"

    #FILTER 2
    # sample_type = "skin"
    # sample_type = "all"
    #FILTER 3
    # not_sample_type = "stool"

    # #FILTER 4
    # human_role = "Partner"
    # human_role = "all"
    #FILTER 5
    # not_human_role = "Offspring"

    #FILTER 6
    # not_familyID = 63


    # # SPLITTING TRAIN and TEST (test = dogs, train = humans)

    # In[5]:


    human_data_train = mergedf[mergedf.host_common_name =="human"]

    dog_data_test = mergedf[mergedf.host_common_name =="dog"]
    dog_data_test.head(1)


    # In[6]:


    # #testing dog forehead to human right palm hypothesis

    # human_data_train = human_data_train[human_data_train.body_site =="UBERON:skin of hand"]
    # print(human_data_train.shape)
    # dog_data_test = dog_data_test[dog_data_test.body_site =="UBERON:face"]
    # print(dog_data_test.shape)


    # In[7]:


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
    # try:
    #     if not_human_role:
    #         human_data_train = human_data_train[human_data_train.role !=not_human_role]    
    # except:
    #     print("no filter here 5")
        
    # try:    
    #     if not_familyID:
    #         human_data_train = human_data_train[human_data_train.role !=not_familyID]
    #         dog_data_test = dog_data_test[dog_data_test.role !=not_familyID]  
    # except:
    #     print("no filter here 6")


    # # Train/test split

    # In[8]:


    X_train = human_data_train.iloc[:, -1035:-1]
    y_train = human_data_train["familyID"].values.reshape(-1, 1)

    y_test = dog_data_test["familyID"].values.reshape(-1, 1)
    X_test = dog_data_test.iloc[:, -1035:-1]


    # # Hyperparameter Selection

    # In[9]:


    n_estimators = 1000

    from sklearn.ensemble import RandomForestClassifier
    rf = RandomForestClassifier(n_estimators=n_estimators)


    # # Non-Scaled Model Fitting

    # In[10]:


    #from non-scaled data
    rf = rf.fit(X_train, y_train.ravel())
    rf.score(X_test, y_test) #geni impurity coefficient

    print(f'Test Score: {rf.score(X_test, y_test)}')
    print(f'Train Score: {rf.score(X_train, y_train)}')


    # # Scaled X Model Fitting

    # In[11]:


    # from sklearn.preprocessing import StandardScaler
    # X_scaler = StandardScaler().fit(X_train)

    # X_train_scaled = X_scaler.transform(X_train)
    # X_test_scaled = X_scaler.transform(X_test)


    # In[12]:


    # from scaled data
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


    # In[16]:


    maxOutput_noDup = maxOutput.drop_duplicates(subset=['Anonymized_Name'], keep="first")
    print(maxOutput_noDup.shape)
    # maxOutput_noDup


    # In[17]:


    maxOutput_C = maxOutput_noDup[maxOutput_noDup.Actual == maxOutput_noDup.Prediction]

    print(maxOutput_C.shape)
    # maxOutput_C


    # In[18]:


    maxOutput_I = maxOutput_noDup[maxOutput_noDup.Actual != maxOutput_noDup.Prediction]

    print(maxOutput_I.shape)
    # maxOutput_I


    # In[19]:


    correct_total = maxOutput_C["Anonymized_Name"].count()
    incorrect_total = maxOutput_I["Anonymized_Name"].count()

    accuracy_maxed_pred = correct_total/(correct_total + incorrect_total)
    print(f' Accuracy of most-predicted (dog) family: {round(accuracy_maxed_pred,3)}')


    return round(accuracy_maxed_pred,3)




