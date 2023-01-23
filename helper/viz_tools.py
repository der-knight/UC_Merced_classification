import matplotlib.pyplot as plt
import seaborn as sns
import logging
import numpy as np
import math
import pandas as pd
plt.rcParams["font.family"] = "Times New Roman"

def plot_history(history):
    """Plot the training and validation loss and accuracy"""

    plt.style.use("ggplot")

    font = { 'size': 15, 'color':  'black', 'weight': 'normal', 'verticalalignment': 'baseline', 'horizontalalignment': 'center'}
    fig, ax = plt.subplots(1,2,figsize=(15,5),dpi=300)
    ax[0].set_title('Loss')
    ax[1].set_title('Accuracy')
    # plot loss
    ax[0].plot(history.history['loss'], label='Train loss')
    ax[0].plot(history.history['val_loss'], label='Test loss')
    ax[0].legend(loc='upper right', prop={'size': 15})
    ax[0].set_xlabel('Epochs', fontsize = 15)
    ax[0].set_ylabel('Loss', fontsize = 15)

    ax[1].plot(history.history['accuracy'], label='Train accuracy')
    ax[1].plot(history.history['val_accuracy'], label='Test accuracy')
    ax[1].legend(loc='lower right', prop={'size': 15})
    ax[1].set_xlabel('Epochs', fontsize = 15)
    ax[1].set_ylabel('Accuracy', fontsize = 15)
    plt.show()


def plot_confusion_matrix(cm, classes):
    """Plot the confusion matrix"""

    df_cm = pd.DataFrame(cm, columns=classes, index = classes)
    df_cm.index.name = 'True Label'
    df_cm.columns.name = 'Predicted Label'
    plt.figure(figsize = (20,15),dpi=600)
    sns.set(font_scale=1.4)#for label size
    sns.heatmap(df_cm, cmap="Blues", annot=True,annot_kws={"size": 16})
    plt.xlabel('Predicted Label', fontsize=30)
    plt.ylabel('True Label', fontsize=30)
    plt.show()


def plot_test_images(images,predicted_label,true_label,class_label):
    """Plot random test images with predicted and actual labels"""
    logger = logging.getLogger()
    old_level = logger.level
    logger.setLevel(100)

    lis=np.random.randint(0,len(images),16)
    img_list=images[lis]

    predicted_label=class_label[predicted_label[lis]]
    actual_label=class_label[true_label[lis]]
    
    fig, ax = plt.subplots(4,4, figsize=(20,20))
    for i in range(4):
        for j in range(4):
            ax[i][j].imshow(img_list[i*4+j])
            ax[i][j].grid(False)
            ax[i][j].set_title('true label:'+ actual_label[i*4+j]+' '+ ' predict label:'+ predicted_label[i*4+j],fontsize=12)
            ax[i][j].xaxis.set_ticks([])
            ax[i][j].yaxis.set_ticks([])

    logger.setLevel(old_level)
    plt.show()

def plot_train_images(train_images,train_labels,class_labels):   
    """Plot random train images with labels for EDA"""
    logger = logging.getLogger()
    old_level = logger.level
    logger.setLevel(100)
    random=np.random.randint(0,train_images.shape[0],16)
    plot_images=train_images[random]
    labels=class_labels[np.argmax(train_labels[random],axis=1)]
    fig,ax=plt.subplots(4,4,figsize=(20,20))
    for i in range(4):
        for j in range(4):
            ax[i,j].imshow(plot_images[i*4+j])
            ax[i,j].set_title(labels[i*4+j])
            ax[i,j].axis('off')
            ax[i,j].grid(False)
    logger.setLevel(old_level)
    plt.show()

def plot_incorrect(images,predicted_label,true_label,class_label):
    """Plot incorrect predictions with predicted and actual labels"""
    logger = logging.getLogger()
    old_level = logger.level
    logger.setLevel(100)
    lis=np.where(predicted_label!=true_label)[0]

    img_list=images[lis]
    predicted_label=class_label[predicted_label[lis]]
    actual_label=class_label[true_label[lis]]
    rows=math.ceil(len(lis)/4)

    fig, ax = plt.subplots(rows,4 ,figsize=(20,rows*5))

    for i in range(rows):
        for j in range(4):
            if(4*i+j<len(lis)):
                ax[i][j].imshow(img_list[i*4+j])
                ax[i][j].grid(False)
                ax[i][j].set_title('true label:'+ actual_label[i*4+j]+' '+ ' predict label:'+ predicted_label[i*4+j],fontsize=10)
                ax[i][j].xaxis.set_ticks([])
                ax[i][j].yaxis.set_ticks([])
            else:
                fig.delaxes(ax[i][j])

    logger.setLevel(old_level)
    plt.show()