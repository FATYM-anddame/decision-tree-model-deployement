import pandas as pd
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import graphviz
from pandas import DataFrame
from sklearn import tree
from sklearn import tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import pickle


def show():
    st.title("Prediction du type d'attaque caradiaque ")
    data = pd.read_csv("heart2.csv")
    var = data.columns
    p_data = data.drop(columns=['num'])
    # categorize the age

    # Add a photo with fixed position
    st.markdown(
        """
    <style>
    .center {
        display: flex;
        justify-content: center;
        align-items: center;
        height: 300px;
    }
    </style>
    """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
    <div class="center">
        <img src='' alt='Photo' width='500'>
    </div>
    """,
        unsafe_allow_html=True
    )

    markdown_text = """
 ## 1 - À propos des données
### 1 - 1 - Description des données
Ce dataset contient des données liées à des patients atteints de maladies cardiaques.
Il vise à aider à la classification et à la prédiction de la présence de maladie cardiaque chez les patients. 
Les caractéristiques incluses dans le dataset fournissent des informations sur différents aspects pertinents 
pour la santé cardiaque des patients.
Elle est souvent représentée comme une caractéristique binaire (1 pour la présence de maladie cardiaque, 0 pour son absence). Dans votre cas, vous avez mentionné que vous recherchez une variable cible à classes multiples, vous pourriez donc avoir plusieurs classes pour représenter différents niveaux de gravité de la maladie cardiaque ou des types spécifiques de maladies cardiaques.
 Le jeu de données est composé de 1000 lignes et comprend 7 caractéristiques différentes, avec une seule variable cible

### Colonnes

***Age*** : L'âge est un facteur important dans les maladies cardiaques, car le risque augmente avec l'âge.
 Il peut être considéré comme une caractéristique numérique.

***CP (Chest Pain Type)*** : C'est une mesure de la douleur ou de l'inconfort ressenti dans la poitrine.

***Trestbps (Resting Blood Pressure)*** : La tension artérielle au repos est une mesure de la pression dans les artères 
lorsque la personne est au repos. Elle est souvent mesurée en millimètres de mercure (mmHg).

***Chol (Cholestérol)*** : Il s'agit du taux de cholestérol présent dans le sang.

***Fbs (Fasting Blood Sugar)*** : Cette mesure indique si la personne a un taux de sucre élevé dans le sang après une période de jeûne.

***Restecg (Résultats Électrocardiographiques au Repos)*** : Il s'agit des résultats d'un test appelé électrocardiogramme(ECG)
qui mesure l'activité électrique du cœur.

***Thalch (Fréquence Cardiaque Maximale Atteinte)*** : C'est la fréquence cardiaque maximale atteinte par la personne 
pendant un exercice physique.

***Exang (Angine d'Effort)*** : L'angine d'effort fait référence à des douleurs ou un malaise thoracique qui se produit pendant 
une activité physique. Elle est souvent représentée comme une caractéristique binaire (1 pour la présence d'angine d'effort, 0 pour son absence).

***Oldpeak (Dépression du Segment ST Induite par l'Exercice par Rapport au Repos)*** : L'oldpeak est une mesure 
de l'abaissement du tracé électrique du cœur dans un enregistrement.

***Slope (Pente du Segment ST à l'Apogée de l'Exercice)*** : La pente mesure comment votre cœur réagit pendant un exercice. 

***Num (Présence de Maladie Cardiaque)*** : La variable cible représente la présence de maladie cardiaque. 
"""
    st.markdown(markdown_text, unsafe_allow_html=True)

    st.write("### Un exemple de données:")
    st.write(data.head(10))
    viz = st.selectbox("Caractéristiques :", var)
    st.write("  ###  1 - 2 -La visualisation des données ")
    fig, ax = plt.subplots()
    sns.set(style="whitegrid")
    sns.countplot(data, x=viz)
    st.pyplot(fig)
    fig, ax = plt.subplots()
    sns.set_color_codes(palette='deep')
    sns.histplot(data, x=viz, hue=data['num'])
    st.pyplot(fig)
    st.markdown("""
    ## 3- Construction du modèle 

    ### 3-1 À propos du modèle


    `Un classificateur d'arbre de décision` est un modèle d'apprentissage automatique qui effectue des prédictions
    en partitionnant les données d'entrée en fonction d'un ensemble de règles de décision déduites des données d'entraînement.
    Chaque nœud de l'arbre représente un point de décision, et les branches représentent les résultats possibles.
    Le modèle est construit en divisant de manière récursive les données en sous-ensembles en fonction de 
    différentes caractéristiques jusqu'à atteindre la profondeur maximale ou une condition d'arrêt.

    ### 3-2 Modèle de classificateur d'arbre de décision
    Le modèle de classificateur d'arbre de décision est créé à l'aide de la classe DecisionTreeClassifier
    dans la bibliothèque scikit-learn. Voici les paramètres les plus couramment utilisés de cette classe :"""
                )

    st.write("###### **DecisionTreeClassifier(criterion='gini', max_depth=None, splitter='best', random_state=None)**")
    st.markdown("""
    Le paramètre  ***`max_depth`*** détermine le nombre maximum de niveaux dans l'arbre de décision,
     limitant ainsi sa complexité et prévenant le surajustement (overfitting).

    Le paramètre ***`random_state`*** contrôle l'aléatoire dans la construction du modèle,
     assurant ainsi la reproductibilité des résultats.

    Le paramètre `splitter` dans un classificateur d'arbre de décision détermine la stratégie utilisée pour diviser les nœuds.
    Deux options sont disponibles : "best" (meilleure) et "random" (aléatoire).

    Enfin, le paramètre `criterion` définit la mesure utilisée pour évaluer la qualité d'une division à chaque nœud,
    telle que l'indice de Gini ou le gain d'information.


    Choisir les bonnes valeurs pour ces paramètres est important pour garantir que le classificateur d'arbre de décision
    fonctionne bien et généralise aux données non vues.
    """)
    test_size = st.slider('test size', 0.10, 0.50, 0.20, key="size")
    splitter = st.radio('splitter', ['best', 'random'], key="splitter")
    max_depth = st.slider('max depth', 1, 50, 4, key="max_depth")
    criterion = st.radio('criterion', ['gini', 'entropy'], key="criterion")
    random_state = st.slider('random state', 0, 42, 25, key="random_state")
    # model building
    target = 'num'
    encode = {'age', 'sex', 'trestbps', 'slope', 'cp', 'chol','restecg', 'thalch', 'oldpeak'}
    #dummy encoding
    for col in encode:
        dummy = pd.get_dummies(data[col], prefix_sep=col)
        data = pd.concat([data, dummy], axis=1)
        del data[col]
    # separate X and Y
    X = data.drop('num', axis=1)
    y = data.num
    # Splitting the dataset into the Training set and Test set
    training_features, test_features, \
        training_target, test_target = train_test_split(X, y, test_size=0.2, random_state=42)
    from sklearn.dummy import DummyClassifier

    # build decision tree model
    DecisionTreeModel = DecisionTreeClassifier(criterion=criterion, random_state=random_state, max_depth=max_depth)
    clf = DecisionTreeModel.fit(training_features, training_target)
    DT_Pred = DecisionTreeModel.predict(test_features)
    # saving the model
    pickle.dump(DecisionTreeModel, open('../adult_DT.pkl', 'wb'))
    # showing the decision tree
    run = st.button("Run", key="run")
    dot_data = tree.export_graphviz(clf, out_file=None, feature_names=X.columns, class_names=['0', '1', '3', '4'],
                                        filled=True,
                                        rounded=True, special_characters=True)
    graph = graphviz.Source(dot_data)
    graph.render("decision_tree")
    if run:
      graph.view()  # Display the visualization in a window
      # decision tree evaluation
      st.write("### evaluation du modéle ")
    val = st.selectbox('metric of evaluation', ['confusion_matrix', 'accuracy'])

    if val == 'confusion_matrix':
          CMDT = confusion_matrix(test_target, DT_Pred)
          CMDT
          fg, x = plt.subplots(figsize=(5, 3))
          sns.heatmap(CMDT, xticklabels=['0', '1', '2', '3', '4'], yticklabels=['0', '1', '2', '3', '4'], annot=True, ax=x,
                     linewidths=0.1,
                     linecolor="Darkblue", cmap="Blues")
          plt.title("confusion Matrix", fontsize=14)
          st.pyplot(fg)
    else:
            ADT = accuracy_score(test_target, DT_Pred)
            st.write(" Decision Tree Prediction Accuracy : {:.2f}%".format(ADT * 100))
    st.write("### la prediction du type de l'attaque cardiaque chez un patient")
    def user_input_features():
        age = st.slider("***l'age du patient***", 28, 70, 45, key="age")
        gender = st.radio("***le sex :***", ["Female", "Male"], key='gender')
        cp = st.radio("***Type de douleur thoracique :***", ["asymptomatic", 'non-anginal', "atypical anggina", "typical angina"], key="cp")
        trestbps = st.radio("***Tension artérielle au repos (Trestbps)***", ["Mild Hypertension", "normal", "Moderate or Severe Hypertentsion"],key="trestbps")
        chol = st.radio("***Niveau de cholestérol (Chol)***", ["high", "Borderline high", "Desirable"], key="chol")
        fbs = st.radio("***Taux de sucre dans le sang à jeun (Fbs)***", ["false", "true"],key="fbs")
        restecg = st.radio("***Résultats électrocardiographiques au repos (Restecg)***", ["NORMAL", "Iv hypertrophy", "st-t abnormality"], key="cg")
        thalch = st.radio("***Fréquence cardiaque maximale atteinte (Thalch)***", ["High", "Medium", "Low", "Very Low"], key="thalch")
        exang = st.radio("***Présence d'angine d'effort (Exang)***", ["false", "true"], key="ex")
        oldpeak = st.radio("***Dépression du segment ST induite par l'exercice (Oldpeak)***", ["Normal", "Abnormal High", "Abnormal Low"], key="old")
        slope = st.radio("***Pente du segment ST à l'apogée de l'exercice (Slope)***", ["flat", "downsloping", "upsloping"], key="slope")
        if fbs == "true":
            fbs = 1
        else:
            fbs = 0
        if exang == "true":
            exang = 1
        else:
            exang = 0

        data = {"age": age, "sex": gender, "cp": cp, "trestbps": trestbps, "chol": chol,
                "fbs": fbs, "restecg": restecg, "thalch": thalch, "exang": exang,
                "oldpeak": oldpeak, "slope": slope}
        features = pd.DataFrame(data, index=[0])
        return features
    input_df = user_input_features()
    bins = [42, 48, 55, 60, 77]
    labels = ['28-42', '42-48', '48-55', '55-70']
    input_df['age'] = pd.cut(input_df['age'], bins=bins, labels=labels)
    encode = {'age', 'sex', 'trestbps', 'slope', 'cp', 'chol', 'restecg', 'thalch', 'oldpeak'}
    p_data = pd.concat([p_data, input_df], axis=0)
    for col in encode:
        dummy = pd.get_dummies(p_data[col], prefix_sep=col)
        p_data = pd.concat([p_data, dummy], axis=1)
        del p_data[col]

    df = p_data.tail(1)
    if st.button('predict', key="predict"):
     load_csf = pickle.load(open('../adult_DT.pkl', 'rb'))
     prediction = load_csf.predict(df)
     prediction_proba = load_csf.predict_proba(df)
     st.subheader("le type de l'attaque cardiaque chez ce patient est :")
     st.write(prediction)
     st.subheader('Probabilité de prediction')
     st.write(prediction_proba)

show()