import warnings
import h5py
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from constants import *

warnings.filterwarnings('ignore')


def training():
    if not os.path.exists(TEST_PATH):
        os.makedirs(TEST_PATH)

    models = [('LR', LogisticRegression(random_state=SEED)),
              ('KNN', KNeighborsClassifier()),
              ('CART', DecisionTreeClassifier(random_state=SEED)),
              ('RF', RandomForestClassifier(n_estimators=NUM_TREES, random_state=SEED)),
              ('NB', GaussianNB())]

    results = []
    names = []

    h5f_data = h5py.File(H5_DATA, 'r')
    h5f_label = h5py.File(H5_LABELS, 'r')

    global_features_string = h5f_data['dataset_1']
    global_labels_string = h5f_label['dataset_1']

    global_features = np.array(global_features_string)
    global_labels = np.array(global_labels_string)

    h5f_data.close()
    h5f_label.close()

    print(f"[STATUS] features shape: {global_features.shape}")
    print(f"[STATUS] labels shape: {global_labels.shape}")
    print("[STATUS] training started...")

    (x_train, x_test, y_train, y_test) = train_test_split(np.array(global_features),
                                                          np.array(global_labels),
                                                          test_size=TEST_SIZE,
                                                          random_state=SEED)

    print("[STATUS] separation train and test data...")
    print(f"Train data  : {x_train.shape}")
    print(f"Test data   : {x_test.shape}")
    print(f"Train labels: {y_train.shape}")
    print(f"Test labels : {y_test.shape}")

    for name, model in models:
        cv_results = cross_val_score(model, x_train, y_train, cv=10, scoring=SCORING, n_jobs=-1)

        model.fit(x_train, y_train)
        print(classification_report(y_test, model.predict(x_test)))
        results.append(cv_results)
        names.append(name)
        print(f'{name}: {cv_results.mean()}')

    fig = plt.figure()
    fig.suptitle('Machine Learning algorithm comparison')
    ax = fig.add_subplot(111)
    plt.boxplot(results)
    ax.set_xticklabels(names)
    plt.savefig('data/result/box.png')
    return x_train, y_train
