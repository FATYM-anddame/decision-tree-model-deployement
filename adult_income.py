import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import graphviz
from sklearn import tree
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import pickle


def show():
    st.title("Prediction de l'income d'un individu")
    data = pd.read_csv("adult_income_v2.csv")
    var = data.columns

    data_p = data

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

    st.image(
        'https://www.hrmagazine.co.uk/media/akufcj5y/class.jpeg?width=960&height=540&quality=80&bgcolor=White&format=webp&rnd=133184228648030000')

    markdown_text = """ 

## 1 - À propos des données
### 1 - 1 - Données générales
<div style="text-align: justify;">
Le jeu de données fournit des informations sur les caractéristiques démographiques et professionnelles des individus, ce qui permet d'analyser les profils et les situations professionnelles afin de comprendre les facteurs qui influencent les niveaux de revenu. Il vise également à classifier le niveau de revenu d'un individu en fonction de s'il est supérieur à 50 000 $ par an ou inférieur. Le jeu de données est composé de 1000 lignes et comprend 7 caractéristiques différentes, avec une seule variable cible binaire appelée "income" qui indique si un individu gagne plus de 50 000 $ par an ou non.
</div>

### 1 - 2 -Caractéristiques 

<div style="text-align: justify;">

**age**: l'âge d'un individu (entier supérieur à 0).

**workclass** : un terme général pour représenter le statut d'emploi d'un individu (Privé, Indépendant sans salaire, Indépendant avec salaire, Gouvernement fédéral, Gouvernement local, Gouvernement d'État, Sans salaire, Jamais travaillé).

**education** : le niveau d'éducation atteint par chaque individu dans l'ensemble de données.

**occupation** : l'emploi actuel ou la profession de chaque individu dans l'ensemble de données.

**marital_status** : l'état matrimonial d'un individu. Marié-civ-spouse correspond à un conjoint civil, tandis que Married-AF-spouse est un conjoint dans les forces armées.

**occupation** : le type général d'occupation d'un individu.

**gender** : le sexe biologique de l'individu (Homme, Femme).

**hours-per-week** : le nombre d'heures qu'un individu a déclaré travailler par semaine (continu).

**income** : si un individu gagne plus de 50 000 $ par an ou non (<= 50K, >50K).
</div>


"""

    st.markdown(markdown_text, unsafe_allow_html=True)

    st.write("### Exemple de jeu de données :")
    st.write(data.head(10))
    st.write("  ###  1 - 2 -La visualisation des données ")
    lst = ['age', 'workclass', 'education',
           'marital_status', 'occupation', 'gender', 'hours_per_week', 'income']
    # visualize the features
    viz = st.selectbox("Quelle colonne souhaitez-vous visualiser ?", lst, key='viz')
    fig: object
    if viz == 'workclass' or viz == 'education' or viz == 'occupation' or viz == 'marital_status':
        fig, ax = plt.subplots()
        sns.set(style="whitegrid")
        sns.husl_palette(as_cmap=True)
        sns.countplot(data, y=viz)
        st.pyplot(fig)
        fig, ax = plt.subplots()
        sns.set_color_codes(palette='deep')
        sns.histplot(data, y=viz, hue=data['income'])
        st.pyplot(fig)

    else:
        fig, ax = plt.subplots()
        sns.set(style="whitegrid")
        sns.countplot(data, x=viz)
        st.pyplot(fig)
        fig, ax = plt.subplots()
        sns.set_color_codes(palette='deep')
        sns.histplot(data, x=viz, hue=data['income'])
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
    # the model's parameters
    test_size = st.slider('**test size**', 0.10, 0.50, 0.20, key="size")
    splitter = st.radio('**splitter**', ['best', 'random'], key="splitter")
    max_depth = st.slider('**max depth**', 1, 50, 4, key="max_depth")
    criterion = st.radio('**criterion**', ['gini', 'entropy'], key="criterion")
    random_state = st.slider('**random state**', 0, 42, 25, key="random_state")
    # model building
    # fiting data and encode categorical data to numerical
    data[['income']] = data[['income']].replace('>=50', '0')
    data[['income']] = data[['income']].replace('<50', '0')
    target = 'income'

    encode = {'gender', 'age', 'marital_status', 'occupation', 'workclass', 'education', 'hours_per_week'}
    #dummy encoding
    for col in encode:
        dummy = pd.get_dummies(data[col], prefix_sep=col)
        data = pd.concat([data, dummy], axis=1)
        del data[col]

    # separate X and Y
    X = data.drop('income', axis=1)
    y = data.income
    # Splitting the dataset into the Training set and Test set
    training_features, test_features, \
        training_target, test_target = train_test_split(X, y, test_size=0.2, random_state=42)

    # build decision tree model
    DecisionTreeModel = DecisionTreeClassifier(criterion=criterion, random_state=random_state, max_depth=max_depth)
    clf = DecisionTreeModel.fit(training_features, training_target)
    DT_Pred = DecisionTreeModel.predict(test_features)

    # saving the model
    pickle.dump(DecisionTreeModel, open('../adult_DT.pkl', 'wb'))

    # showing the decision tree
    run = st.button("Run", key="run")
    if run:
        dot_data = tree.export_graphviz(clf, out_file=None, feature_names=X.columns, class_names=['0', '1'],
                                        filled=True,
                                        rounded=True, special_characters=True)
        graph = graphviz.Source(dot_data)
        graph.render("decision_tree")
        graph.view()  # Display the visualization in a window





    # decision tree evaluation
    st.write("### evaluation du modéle ")
    val = st.radio('metric of evaluation', ['accuracy', 'confusion_matrix'])
    ADT = accuracy_score(test_target, DT_Pred)
    CMDT = confusion_matrix(test_target, DT_Pred)
    (CMDT / 200) * 100
    CMDT = pd.crosstab(test_target, DT_Pred, rownames=['Actual'], colnames=['Predicted'])
    fig, ax = plt.subplots()
    sns.heatmap((CMDT / 200) * 100, xticklabels=[">50k", "<=50k"], yticklabels=[">50k", "<=50k"], annot=True, ax=ax,
                    linewidths=0.2,
                    linecolor="Darkblue", cmap="Blues")
    plt.title("Matrice de confusion", fontsize=14)
    st.pyplot(fig)

    if val == 'accuracy':
           st.write(" Decision Tree Prediction Accuracy : {:.2f}%".format(ADT * 100))





    # decision tree predection
    st.write("""
### Prediction de l'income d'un individu
""")

    # collects user input features into dataframe
    def user_input_features():
        occupation = st.selectbox("What is your current occupation?",
                                  ["Exec-managerial", "Prof-specialty", "Craft-repair", "Other-service", "Adm-clerical",
                                   "Sales", "Machine-op-inspct", "Handlers-cleaners", "Transport-moving",
                                   "Farming-fishing",
                                   "Tech-support", "Protective-serv", "Priv-house-serv", "Armed-Forces"], key="occup")

        gender = st.selectbox("**Gender**", ["Female", "Male"], key='gender')
        workclass = st.selectbox("**Workclass**", ["Private", "non-private"], key="work")
        education = st.selectbox("**What is the highest level of education you have completed?**",
                                 ['Less than High School', 'HS-grad', "Some College or Associate's Degree",
                                  "Bachelor's Degree", "Master's Degree", "Professional Degree", "Doctorate"],
                                 key="edu")
        hours_per_week = st.slider("**On average, how many hours do you work per week?**", 1, 99, key="week")
        marital_status = st.selectbox("**Marital Status**",
                                      ['Never-married', 'Married-civ-spouse', 'Widowed', 'Divorced', 'Separated',
                                       "Married-spouse-absent", "Married-AF-spouse"], key="mari")
        age = st.slider("How old are you?", 17, 90, 45, key="age")

        data = {'education': education,
                'gender': gender,
                'occupation': occupation,
                'workclass': workclass,
                'hours_per_week': hours_per_week,
                'age': age,
                'marital_status': marital_status,
                }
        features = pd.DataFrame(data, index=[0])
        return features

    age_bins = [0, 18, 25, 44, 60, 75, 90]
    age_labels = ['<18', '18-25', '25-44', '44-60', '60-75', '75-90']

    h_bins = [0, 7, 20, 35, 40, 60, 99]
    h_labels = ['<8', '8-20', '20-35', '35-40', '40-60', '60-99']
    # get the user input data
    input_df = user_input_features()
    # encoding the user's data
    input_df['age'] = pd.cut(input_df['age'], bins=age_bins, labels=age_labels, right=False)
    input_df['hours_per_week'] = pd.cut(input_df['hours_per_week'], bins=h_bins, labels=h_labels, right=False)
    data_p[['income']] = data_p[['income']].replace('>=50', '1')
    data_p[['income']] = data_p[['income']].replace('<50', '0')
    p_data = data_p.drop(columns=['income'])
    p_data = pd.concat([p_data, input_df], axis=0)

    encode = ['education', 'gender', 'occupation', 'workclass', 'hours_per_week', 'age', 'marital_status']
    #dummy encoding
    for coln in encode:
        dummy = pd.get_dummies(p_data[coln], prefix_sep=coln)
        # dummy = dummy.reindex(columns=data[coln], fill_value=0)
        p_data = pd.concat([p_data, dummy], axis=1)
        del p_data[coln]

    p_data = p_data.reindex(columns=X.columns)

    df = p_data.tail(1)
    # make a prediction by the input data
    if st.button('predire', key="predict"):
        st.subheader('user input features')
        load_csf = pickle.load(open('../adult_DT.pkl', 'rb'))
        prediction = load_csf.predict(df)
        prediction_proba = load_csf.predict_proba(df)
        st.subheader("le revenue de cet individu sera  :")
        st.write(prediction)
        st.subheader('Probabilité de prediction ')
        st.write(prediction_proba)
show()