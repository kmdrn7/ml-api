import mysql.connector
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional

app = FastAPI()

db = mysql.connector.connect(
    host="localhost",
    user="alinamed",
    password="alinamed",
    database="ta-finish",
    # database="ml-server",
    auth_plugin="mysql_native_password",
)
db.autocommit = True
cur = db.cursor()

class ConfussionMatrix(BaseModel):
  tn: float
  fp: float
  fn: float
  tp: float

class Metrics(BaseModel):
  code: str
  algorithm: str
  dataset: str
  datatest: Optional[str]
  params: Optional[str]
  matrix: Optional[ConfussionMatrix]
  accuracy: Optional[float]
  balanced_accuracy: Optional[float]
  recall: Optional[float]
  f1: Optional[float]
  precision: Optional[float]
  roc_auc: Optional[float]
  time: Optional[float]

class MetricsRealtime(BaseModel):
  algorithm: str
  dataset: str
  matrix: ConfussionMatrix
  accuracy: float
  balanced_accuracy: float
  recall: float
  f1: float
  precision: float
  roc_auc: float
  time: Optional[float]

# class Model(BaseModel)

@app.get("/api/v1/model/{model_id}")
def get_model_by_id(model_id: str):
  sql = "SELECT * FROM model WHERE id = {}".format(model_id)
  cur.execute(sql)
  result = cur.fetchone()
  if result == None:
    return {
      'data': None,
    }
  else:
    return {
      'data': {
        'id': result[0],
        'features': result[1],
        'joblib': result[2],
        'scaler': result[3],
      }
    }

@app.post("/api/v1/prediction")
def post_prediction(metric: Metrics):
  sql = "INSERT INTO prediction (kode_percobaan, ml, ds, tp, tn, fp, fn, acc, bacc, recall, f1, prec, roc_auc) VALUES ('{}', '{}', '{}', '{}', '{}', {}, {}, {}, {}, {}, {}, {}, {})" .format(
    metric.code, metric.algorithm, metric.dataset,
    metric.matrix.tp, metric.matrix.tn,
    metric.matrix.fp, metric.matrix.fn,
    float(metric.accuracy), float(metric.balanced_accuracy),
    float(metric.recall), float(metric.f1),
    float(metric.precision), float(metric.roc_auc)
  )
  cur.execute(sql)
  return True

@app.post("/api/v1/training")
def post_prediction(metric: Metrics):
  sql = "INSERT INTO training (kode_percobaan, ml, ds, tp, tn, fp, fn, acc, bacc, recall, f1, prec, roc_auc, time) VALUES ('{}', '{}', '{}', '{}', '{}', {}, {}, {}, {}, {}, {}, {}, {}, {})" .format(
    metric.code, metric.algorithm, metric.dataset,
    metric.matrix.tp, metric.matrix.tn,
    metric.matrix.fp, metric.matrix.fn,
    float(metric.accuracy), float(metric.balanced_accuracy),
    float(metric.recall), float(metric.f1),
    float(metric.precision), float(metric.roc_auc),
    float(metric.time)
  )
  cur.execute(sql)
  return True

@app.post("/api/v1/testing")
def post_testing(metric: Metrics):
  sql = "INSERT INTO testing_model (kode_percobaan, ml, ds, dt, tp, tn, fp, fn, acc, bacc, recall, f1, prec, roc_auc, time) VALUES ('{}', '{}', '{}', '{}', '{}', '{}', {}, {}, {}, {}, {}, {}, {}, {}, {})" .format(
    metric.code, metric.algorithm, metric.dataset, metric.datatest,
    metric.matrix.tp, metric.matrix.tn,
    metric.matrix.fp, metric.matrix.fn,
    float(metric.accuracy), float(metric.balanced_accuracy),
    float(metric.recall), float(metric.f1),
    float(metric.precision), float(metric.roc_auc),
    float(metric.time)
  )
  cur.execute(sql)
  return True

@app.post("/api/v1/model_performance")
def post_model_performance(metric: Metrics):
  sql = "INSERT INTO model_performance (kode, ds, ml, acc, bacc, recall, f1, prec, roc_auc, time) VALUES ('{}', '{}', '{}', '{}', {}, {}, {}, {}, {}, {})" .format(
    metric.code, metric.dataset, metric.algorithm,
    float(metric.accuracy), float(metric.balanced_accuracy),
    float(metric.recall), float(metric.f1),
    float(metric.precision), float(metric.roc_auc),
    float(metric.time)
  )
  cur.execute(sql)
  return True

@app.post("/api/v1/realtime")
def post_realtime(metric: MetricsRealtime):
  sql = "INSERT INTO testing_realtime (model, ds, tp, tn, fp, fn, acc, bacc, recall, f1, prec, roc_auc, time) VALUES ('{}', '{}','{}', '{}', '{}', '{}', {}, {}, {}, {}, {}, {}, {})" .format(
    metric.algorithm, metric.dataset, metric.matrix.tp,
    metric.matrix.tn, metric.matrix.fp, metric.matrix.fn,
    float(metric.accuracy), float(metric.balanced_accuracy),
    float(metric.recall), float(metric.f1),
    float(metric.precision), float(metric.roc_auc), float(metric.time)
  )
  cur.execute(sql)
  return True

@app.post("/api/v1/hyperparameter")
def hyperparameter(metric: Metrics):
  sql = "INSERT INTO hyperparameter_search (kode, ds, ml, params, time) VALUES ('{}', '{}','{}', '{}', '{}')" .format(
    metric.code, metric.dataset, metric.algorithm,
    metric.params, float(metric.time)
  )
  cur.execute(sql)
  return True
