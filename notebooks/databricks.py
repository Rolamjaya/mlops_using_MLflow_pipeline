# Databricks notebook source
# MAGIC
# MAGIC %md
# MAGIC # MLflow Regression Pipeline Databricks Notebook
# MAGIC This notebook runs the MLflow Regression Pipeline on Databricks and inspects its results.
# MAGIC
# MAGIC For more information about the MLflow Regression Pipeline, including usage examples,
# MAGIC see the [Regression Pipeline overview documentation](https://mlflow.org/docs/latest/pipelines.html#regression-pipeline)
# MAGIC and the [Regression Pipeline API documentation](https://mlflow.org/docs/latest/python_api/mlflow.pipelines.html#module-mlflow.pipelines.regression.v1.pipeline).

# COMMAND ----------

# MAGIC %pip install mlflow[pipelines]
# MAGIC %pip install -r ../requirements.txt
# MAGIC !pip install hyperopt
# MAGIC !pip install codecarbon

# COMMAND ----------

! codecarbon init

# COMMAND ----------

# MAGIC %md ### Create a new pipeline with "databricks" profile:

# COMMAND ----------

from mlflow.pipelines import Pipeline
import mlflow
from codecarbon import EmissionsTracker
tracker = EmissionsTracker()
tracker.start()
p = Pipeline(profile="databricks")

# COMMAND ----------

# MAGIC %md ### Inspect a newly created pipeline using a graphical representation:

# COMMAND ----------

p.clean()

# COMMAND ----------

p.inspect()

# COMMAND ----------

# MAGIC %md ### Ingest the dataset into the pipeline:

# COMMAND ----------

#tracker.start()
p.run("ingest")
#tracker.stop()

# COMMAND ----------

# MAGIC %md ### Split the dataset in train, validation and test data profiles:

# COMMAND ----------

#tracker.start()
p.run("split")
#tracker.stop()

# COMMAND ----------

training_data = p.get_artifact("training_data")
training_data.describe()

# COMMAND ----------

#tracker.start()
p.run("transform")
#tracker.stop()

# COMMAND ----------

training_data = p.get_artifact("training_data")
validation_data = p.get_artifact("validation_data")
test_data = p.get_artifact("test_data")
import pandas as pd
data = pd.DataFrame([training_data, validation_data, test_data])
data.write.format("delta").mode("overwrite").option("mergeSchema", "true").save("dbfs:/user/hive/warehouse/athletes_copy")

# COMMAND ----------

# MAGIC %md ### Using training data profile, train the model:

# COMMAND ----------

#tracker.start()
p.run("train")
#tracker.stop()

# COMMAND ----------

trained_model = p.get_artifact("model")
print(trained_model)

# COMMAND ----------

# MAGIC %md ### Evaluate the resulting model using validation data profile:

# COMMAND ----------

#tracker.start()
p.run("evaluate")
#tracker.stop()

# COMMAND ----------

# MAGIC %md ### Register the trained model in the registry:

# COMMAND ----------

#tracker.start()
p.run("register")
tracker.stop()

# COMMAND ----------

# MAGIC %md ### Carbon Emission

# COMMAND ----------

import pandas as pd
carbon_emission = pd.read_csv("/Workspace/Repos/rolamjayahs@gmail.com/mlops_using_MLflow_pipeline/notebooks/emissions.csv")

# COMMAND ----------

display(carbon_emission)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Build chart to compare the model run

# COMMAND ----------

import mlflow
run_id = "e261a22a75a046f69214f954fc1c14f1"
run = mlflow.get_run(run_id)
experiment_name = mlflow.get_experiment(run.info.experiment_id).name

print("Experiment Name:", experiment_name)

# COMMAND ----------

# Get the experiment ID based on the experiment name
experiment = mlflow.get_experiment_by_name(experiment_name)

if experiment:
    experiment_id = experiment.experiment_id
    print("Experiment ID:", experiment_id)
else:
    print("Experiment not found.")

# COMMAND ----------

runs = mlflow.search_runs(experiment_ids=experiment_id)
runs.head(10)

# COMMAND ----------

runs = mlflow.search_runs(experiment_ids=experiment_id,
                          order_by=['metrics.mae'])#, max_results=1)
runs.loc[1]

# COMMAND ----------

from datetime import datetime, timedelta

earliest_start_time = (datetime.now() - timedelta(days=14)).strftime('%Y-%m-%d')
#recent_runs = runs[runs.start_time >= earliest_start_time]
recent_runs = runs[runs["metrics.root_mean_squared_error_on_data_validation"]<2]
#recent_runs = recent_runs.dropna()
#recent_runs['Run Date'] = recent_runs.start_time.dt.floor(freq='D')

#best_runs_per_day_idx = recent_runs.groupby(
#  ['Run Date']
#)['metrics.training_mae'].idxmin()
#best_runs = recent_runs.loc[best_runs_per_day_idx]

display(recent_runs[['run_id', 'metrics.root_mean_squared_error_on_data_validation']])


# COMMAND ----------

import matplotlib.pyplot as plt
plt.figure(figsize=(10,6))
plt.plot(recent_runs['run_id'], recent_runs['metrics.root_mean_squared_error_on_data_validation'])
plt.xlabel('run_id')
plt.ylabel('metrics.root_mean_squared_error_on_data_validation')
plt.xticks(rotation=90)
plt.show()
