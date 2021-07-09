import pandas as pd
import numpy as np
from glob import glob
from joblib import dump, load
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score, balanced_accuracy_score, roc_auc_score, confusion_matrix

def get_csv(uri):
    csvs = []
    if "*" in uri:
        all_csv = glob(uri)
        [ csvs.append(pd.read_csv(uri)) for uri in all_csv ]
        return pd.concat(csvs)
    return pd.read_csv(uri)
cols = ['Label', 'SYN Flag Count', 'Fwd Seg Size Min', 'FWD Init Win Bytes',
       'FIN Flag Count', 'Average Packet Size', 'Packet Length Mean',
       'Packet Length Max', 'Protocol', 'Idle Max', 'Idle Mean', 'Idle Min',
       'Flow Duration', 'Fwd IAT Total', 'Fwd Packet Length Max',
       'Fwd Segment Size Avg', 'Fwd Packet Length Mean',
       'Bwd Packet Length Mean', 'Bwd Segment Size Avg', 'Packet Length Std',
       'Bwd IAT Total', 'Bwd Packet Length Max']
models = [
    {"model": "RandomForest"},
]
collection = [
    {"dataset": "Unseen Dataset", "path": "/media/kmdr7/Seagate/TA/DATASETS/newUnseenDataset.csv", "type": -1},
    {"dataset": "Benign IoTTT *", "path": "/media/kmdr7/Seagate/DATASETS/IoT-Traffic-Traces/out/*", "type": 0},
]
datates = get_csv(collection[0]["path"])[cols]
joblib = "/media/kmdr7/Seagate/TA/MODELS/LogisticRegression.joblib"
clf = load(joblib)
scaler = MinMaxScaler()

y_real = []
y_pred = []
for i in datates.index:
    single = pd.DataFrame([datates.iloc[i]])
    X = single.drop(columns=["Label"])
    y = single["Label"]
    X = X.to_numpy()[0][:, np.newaxis]
    X = pd.DataFrame(scaler.fit_transform(X))
    X = np.transpose(X.values)
    pred = clf.predict(X)
    y_real.append(int(y.values))
    y_pred.append(int(pred[0]))

tn, fp, fn, tp = confusion_matrix(y_real, y_pred, labels=[0,1]).ravel()
print("True Positive", tp)
print("True Negative", tn)
print("False Positive", fp)
print("False Negative", fn)

acc = accuracy_score(y, pred)
print("Accuracy", acc)

bacc = balanced_accuracy_score(y, pred)
print("Balanced Accuracy", bacc)

recall = recall_score(y, pred)
print("Recall", recall)

f1 = f1_score(y, pred)
print("F1", f1)

precision = precision_score(y, pred)
print("Precission", precision)

try: roc_auc = roc_auc_score(y, pred)
except: roc_auc = 0
print("Roc AUC", roc_auc)
