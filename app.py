import streamlit as st

import scipy
from mat4py import loadmat
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn import metrics





st.title('Hyperspectral Image Classification')
Dataset = st.sidebar.selectbox('select Dataset',('Indian pines','pavia university','salinas'))
classify = st.sidebar.selectbox('select classifier',('SVM','decisiontree','KNeighborsClassifier','RandomForestClassifier'))
st.balloons()



def main():
    if Dataset == 'Indian pines':
        x = loadmat('Indian_pines_corrected.mat')['indian_pines_corrected']
        y = loadmat('Indian_pines_gt.mat')['indian_pines_gt']
        ys = y.shape
        names = ['Alfalfa', 'Corn-notill', 'Corn-mintill', 'Corn'
                        ,'Grass-pasture', 'Grass-trees', 'Grass-pasture-mowed',
                        'Hay-windrowed', 'Oats', 'Soybean-notill', 'Soybean-mintill',
                        'Soybean-clean', 'Wheat', 'Woods', 'Buildings-Grass-Trees-Drives',
                        'Stone-Steel-Towers']
    elif Dataset == 'pavia university':
        x = loadmat('PaviaU.mat')['paviaU']
        y = loadmat('PaviaU_gt.mat')['paviaU_gt']
        ys = y.shape
        names = ['Asphalt', 'Meadows', 'Gravel', 'Trees', 'Painted metal sheets', 'Bare Soil', 'Bitumen',
                 'Self-Blocking Bricks', 'Shadows']
    elif Dataset == 'salinas':
        x = loadmat('Salinas_corrected.mat')['salinas_corrected']
        y = loadmat('Salinas_gt.mat')['salinas_gt']
        ys = y.shape
        names = ['Brocoli_green_weeds_1','Brocoli_green_weeds_2','Fallow','Fallow_rough_plow','Fallow_smooth',
                        'Stubble','Celery','Grapes_untrained','Soil_vinyard_develop','Corn_senesced_green_weeds',
                        'Lettuce_romaine_4wk','Lettuce_romaine_5wk','Lettuce_romaine_6wk','Lettuce_romaine_7wk',
                        'Vinyard_untrained','Vinyard_vertical_trellis']
    fig = plt.figure(figsize=(12, 6))
    q = np.random.randint(x.shape[2])
    plt.imshow(x[:, :, q], cmap='nipy_spectral')
    plt.axis('off')
    plt.title(f'Band - {q}')
    st.pyplot()
    plt.figure(figsize=(12, 6))
    plt.imshow(y, cmap='jet')
    plt.colorbar()
    plt.axis('off')
    #plt.show()
    st.title('ground_truth')
    st.pyplot()
    q = x.reshape(-1, x.shape[2])
    df = pd.DataFrame(data=q)
    df = pd.concat([df, pd.DataFrame(data=y.ravel())], axis=1)
    df.columns = [f'band{i}' for i in range(1, 1 + x.shape[2])] + ['class']
    df.to_csv('hssi.csv')
    df2 = pd.read_csv('hssi.csv')
    del df2['Unnamed: 0']
    pca = PCA(n_components=40)
    dt = pca.fit_transform(df2.iloc[:, :-1].values)
    r = pd.concat([pd.DataFrame(data=dt), pd.DataFrame(data=y.ravel())], axis=1)
    r.columns = [f'PC-{i}' for i in range(1, 41)] + ['class']
    r.head()
    r.to_csv('hssi_after_pca.csv')
    fig = plt.figure(figsize=(12, 6))
    q = np.random.randint(x.shape[2])
    plt.imshow(x[:, :, q], cmap='nipy_spectral')
    plt.axis('off')
    plt.title(f'Band - {q}')
    f = r[r['class'] != 0]
    X = f.iloc[:, :-1].values
    y = f.loc[:, 'class'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=11, stratify=y)
    if classify == 'SVM':
        svm = SVC(C=100, kernel='rbf', cache_size=10 * 1024)
        svm.fit(X_train, y_train)
        ypred = svm.predict(X_test)
        data = confusion_matrix(y_test, ypred)
        df_cm = pd.DataFrame(data, columns=np.unique(names), index=np.unique(names))
        df_cm.index.name = 'Actual'
        df_cm.columns.name = 'Predicted'
        plt.figure(figsize=(12, 6))
        sns.set(font_scale=1.4)
        st.title('Confusion Matrix')
        sns.heatmap(df_cm, cmap="Reds", annot=True, annot_kws={"size": 16}, fmt='d')
        st.write('Confusion Matrix')
        st.pyplot()
        accuracy = classification_report(y_test, ypred, target_names=names)
        st.write(accuracy)
        l = []
        for i in range(r.shape[0]):
            if r.iloc[i, -1] == 0:
                l.append(0)
            else:
                l.append(svm.predict(r.iloc[i, :-1].values.reshape(1, -1)))
        clmap = np.array(l).reshape(ys[0],ys[1], ).astype('float')
        plt.figure(figsize=(12, 6))
        plt.imshow(clmap, cmap='nipy_spectral')
        plt.colorbar()
        plt.axis('off')
        st.title('Classification Map')
        st.pyplot()
        plt.figure(figsize=(12, 6))
        pixel_no = np.random.randint(df2.shape[0])
        plt.plot(range(1, 201), df2.iloc[pixel_no, :-1].values.tolist(), 'b--', label=f'Class - {df2.iloc[pixel_no, -1]}')
        plt.legend()
        plt.title(f'Pixel({pixel_no}) signature', fontsize=14)
        plt.xlabel('Band Number', fontsize=14)
        plt.ylabel('Pixel Intensity', fontsize=14)
        st.pyplot()
    elif classify == 'decisiontree':
        model = DecisionTreeClassifier()
        model = model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        data = confusion_matrix(y_test, y_pred)
        df_cm = pd.DataFrame(data, columns=np.unique(names), index=np.unique(names))
        df_cm.index.name = 'Actual'
        df_cm.columns.name = 'Predicted'
        plt.figure(figsize=(12, 6))
        sns.set(font_scale=1.4)
        st.title('Confusion Matrix')
        sns.heatmap(df_cm, cmap="Reds", annot=True, annot_kws={"size": 16}, fmt='d')
        st.pyplot()
        st.write("Accuracy:", metrics.accuracy_score(y_test, y_pred))
        l2 = []
        for i in range(r.shape[0]):
            if r.iloc[i, -1] == 0:
                l2.append(0)
            else:
               l2.append(model.predict(r.iloc[i, :-1].values.reshape(1, -1)))
        cl2map = np.array(l2).reshape(ys[0],ys[1], ).astype('float')
        plt.figure(figsize=(12, 6))
        plt.imshow(cl2map, cmap='nipy_spectral')
        plt.colorbar()
        plt.axis('off')
        st.title('Classification Map')
        st.pyplot()
        accuracy = classification_report(y_test, y_pred, target_names=names)
        st.write(accuracy)
    elif classify == 'KNeighborsClassifier':
        model = KNeighborsClassifier(n_neighbors=5)
        model = model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        st.write("Accuracy:", metrics.accuracy_score(y_test, y_pred))
        data = confusion_matrix(y_test, y_pred)
        df_cm = pd.DataFrame(data, columns=np.unique(names), index=np.unique(names))
        df_cm.index.name = 'Actual'
        df_cm.columns.name = 'Predicted'
        plt.figure(figsize=(12, 6))
        sns.set(font_scale=1.4)
        st.title('Confusion Matrix')
        sns.heatmap(df_cm, cmap="Reds", annot=True, annot_kws={"size": 16}, fmt='d')
        st.pyplot()
        l3 = []
        for i in range(r.shape[0]):
            if r.iloc[i, -1] == 0:
                l3.append(0)
            else:
                l3.append(model.predict(r.iloc[i, :-1].values.reshape(1, -1)))
        cl3map = np.array(l3).reshape(ys[0],ys[1], ).astype('float')
        plt.figure(figsize=(12, 6))
        plt.imshow(cl3map, cmap='nipy_spectral')
        plt.colorbar()
        plt.axis('off')
        st.title('Classification Map')
        st.pyplot()
        accuracy = classification_report(y_test, y_pred, target_names=names)
        st.write(accuracy)
    elif classify == 'RandomForestClassifier':
        model3 = RandomForestClassifier(n_estimators=100)
        model3.fit(X_train, y_train)
        y_pred = model3.predict(X_test)
        st.write("Accuracy:", metrics.accuracy_score(y_test, y_pred))
        data = confusion_matrix(y_test, y_pred)
        df_cm = pd.DataFrame(data, columns=np.unique(names), index=np.unique(names))
        df_cm.index.name = 'Actual'
        df_cm.columns.name = 'Predicted'
        plt.figure(figsize=(12, 6))
        sns.set(font_scale=1.4)
        st.title('Confusion Matrix')
        sns.heatmap(df_cm, cmap="Reds", annot=True, annot_kws={"size": 16}, fmt='d')
        st.pyplot()
        l4 = []
        for i in range(r.shape[0]):
            if r.iloc[i, -1] == 0:
                l4.append(0)
            else:
                l4.append(model3.predict(r.iloc[i, :-1].values.reshape(1, -1)))
        cl4map = np.array(l4).reshape(ys[0],ys[1],).astype('float')
        plt.figure(figsize=(12, 6))
        plt.imshow(cl4map, cmap='nipy_spectral')
        plt.colorbar()
        plt.axis('off')
        st.title('Classification Map')
        st.pyplot()
        st.write(classification_report(y_test, y_pred, target_names=names))

if st.sidebar.button('classify'):
        main()
