import glob
import pandas as pd
import numpy as np
from sklearn import preprocessing
from tensorflow.keras.applications.mobilenet import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.utils import to_categorical

def get_files():
    files=glob.glob('./data/UCMerced_LandUse/UCMerced_LandUse/Images/*/*')
    label=files[0].split('\\')[-2]
    df=pd.DataFrame()
    df['files_location']=files
    df['label']=df['files_location'].apply(lambda x:x.split('\\')[-2])
    le = preprocessing.LabelEncoder()
    le.fit(df['label'].unique())
    df['encoded_label']=le.transform(df['label'])

    subset_labels=df['label'].unique()[:14]
    subset_df=df[df['label'].isin(subset_labels)]
    alternate_df=df[~df['label'].isin(subset_labels)].reset_index(drop=True)

    subset_df=subset_df.sample(frac=1).reset_index(drop=True)
    train_df=pd.DataFrame()
    test_df=pd.DataFrame()
    val_df=pd.DataFrame()   
    for labels in subset_labels:
        df_label_train=subset_df[subset_df['label']==labels][0:70]
        df_label_val=subset_df[subset_df['label']==labels][70:80]
        df_label_test=subset_df[subset_df['label']==labels][80:]
        train_df=pd.concat([train_df,df_label_train],axis=0).reset_index(drop=True)
        test_df=pd.concat([test_df,df_label_test],axis=0).reset_index(drop=True)
        val_df=pd.concat([val_df,df_label_val],axis=0).reset_index(drop=True)
    return(train_df,test_df,val_df,alternate_df,subset_labels)


def load_data(dataframe):
    images, labels = [], []
        
    for i in range(dataframe.shape[0]):
        
        img_name = dataframe.loc[i,'files_location']
        img = load_img(img_name,target_size=(224,224))
        img = img_to_array(img)
        
        img = preprocess_input(img)
        label = dataframe.loc[i,'encoded_label']
        
        images.append(img)
        labels.append(label)

    images = np.array(images)
    labels = np.array(labels)
    labels = to_categorical(labels)
    return images, labels



