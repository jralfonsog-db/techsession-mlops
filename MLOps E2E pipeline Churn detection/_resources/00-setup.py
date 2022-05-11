# Databricks notebook source
dbutils.widgets.dropdown("reset_all_data", "false", ["true", "false"], "Reset all data")

# COMMAND ----------

# MAGIC %run ../../../_resources/00-global-setup $reset_all_data=$reset_all_data $db_prefix=retail

# COMMAND ----------

spark.sql("create table if not exists bronze_customers location '/mnt/field-demos/retail/customer_churn'")

# COMMAND ----------

from pyspark.sql.functions import col
from databricks.feature_store import FeatureStoreClient
import mlflow

import databricks
from databricks import automl
from datetime import datetime

def get_automl_run(name):
  #get the most recent automl run
  df = spark.table("field_demos_metadata.automl_experiment").filter(col("name") == name).orderBy(col("date").desc()).limit(1)
  return df.collect()

#Get the automl run information from the field_demos_metadata.automl_experiment table. 
#If it's not available in the metadata table, start a new run with the given parameters
def get_automl_run_or_start(name, model_name, dataset, target_col, timeout_minutes):
  spark.sql("create database if not exists field_demos_metadata")
  spark.sql("create table if not exists field_demos_metadata.automl_experiment (name string, date string)")
  result = get_automl_run(name)
  if len(result) == 0:
    print("No run available, start a new Auto ML run, this will take a few minutes...")
    start_automl_run(name, model_name, dataset, target_col, timeout_minutes)
    result = get_automl_run(name)
  return result[0]


#Start a new auto ml classification task and save it as metadata.
def start_automl_run(name, model_name, dataset, target_col, timeout_minutes = 5):
  automl_run = databricks.automl.classify(
    dataset = dataset,
    target_col = target_col,
    timeout_minutes = timeout_minutes
  )
  experiment_id = automl_run.experiment.experiment_id
  path = automl_run.experiment.name
  data_run_id = mlflow.search_runs(experiment_ids=[automl_run.experiment.experiment_id], filter_string = "tags.mlflow.source.name='Notebook: DataExploration'").iloc[0].run_id
  exploration_notebook_id = automl_run.experiment.tags["_databricks_automl.exploration_notebook_id"]
  best_trial_notebook_id = automl_run.experiment.tags["_databricks_automl.best_trial_notebook_id"]

  cols = ["name", "date", "experiment_id", "experiment_path", "data_run_id", "best_trial_run_id", "exploration_notebook_id", "best_trial_notebook_id"]
  spark.createDataFrame(data=[(name, datetime.today().isoformat(), experiment_id, path, data_run_id, automl_run.best_trial.mlflow_run_id, exploration_notebook_id, best_trial_notebook_id)], schema = cols).write.mode("append").option("mergeSchema", "true").saveAsTable("field_demos_metadata.automl_experiment")
  #Create & save the first model version in the MLFlow repo (required to setup hooks etc)
  mlflow.register_model(f"runs:/{automl_run.best_trial.mlflow_run_id}/model", model_name)
  return get_automl_run(name)

#Generate nice link for the given auto ml run
def display_automl_link(name, model_name, dataset, target_col, timeout_minutes = 5):
  r = get_automl_run_or_start(name, model_name, dataset, target_col, timeout_minutes)
  html = f"""For exploratory data analysis, open the <a href="/#notebook/{r["exploration_notebook_id"]}">data exploration notebook</a><br/><br/>"""
  html += f"""To view the best performing model, open the <a href="/#notebook/{r["best_trial_notebook_id"]}">best trial notebook</a><br/><br/>"""
  html += f"""To view details about all trials, navigate to the <a href="/#mlflow/experiments/{r["experiment_id"]}/s?orderByKey=metrics.%60val_f1_score%60&orderByAsc=false">MLflow experiment</>"""
  displayHTML(html)


def display_automl_churn_link(): 
  display_automl_link("churn_auto_ml", "field_demos_customer_churn", spark.table("churn_features"), "churn", 5)

def get_automl_churn_run(): 
  return get_automl_run_or_start("churn_auto_ml", "field_demos_customer_churn", spark.table("churn_features"), "churn", 5)

# COMMAND ----------

# Replace this with your Slack webhook
slack_webhook = ""


# COMMAND ----------

# MAGIC %run ./API_Helpers

# COMMAND ----------

#https://e2-demo-west.cloud.databricks.com/?o=2556758628403379#mlflow/models?nameSearchInput=field_demos_customer_churn