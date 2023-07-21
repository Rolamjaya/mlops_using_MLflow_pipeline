"""
This module defines custom metric functions that are invoked during the 'train' and 'evaluate'
steps to provide model performance insights. Custom metric functions defined in this module are
referenced in the ``metrics`` section of ``pipeline.yaml``, for example:

.. code-block:: yaml
    :caption: Example custom metrics definition in ``pipeline.yaml``

    metrics:
      custom:
        - name: weighted_mean_squared_error
          function: weighted_mean_squared_error
          greater_is_better: False
"""
from typing import Dict

from pandas import DataFrame


def get_custom_metrics(
    eval_df: DataFrame,
    builtin_metrics: Dict[str, int],  # pylint: disable=unused-argument
) -> Dict[str, int]:
    """
    FIXME::OPTIONAL: provide function doc string.
    :param eval_df: A Pandas DataFrame containing the following columns:
                    - ``"prediction"``: Predictions produced by submitting input data to the model.
                    - ``"target"``: Ground truth values corresponding to the input data.
    :param builtin_metrics: A dictionary containing the built-in metrics that are calculated
                            automatically during model evaluation. The keys are the names of the
                            metrics and the values are the scalar values of the metrics. For more
                            information, see
                            https://mlflow.org/docs/latest/python_api/mlflow.html#mlflow.evaluate.
    :return: A single-entry dictionary containing the custom metrics. The key is the metric name
             and the value is the scalar metric value. Note that custom metric functions can
             return dictionaries with multiple metric entries as well.
    """
    # FIXME::OPTIONAL: implement custom metrics calculation here.

    raise NotImplementedError

def example_custom_metric_fn(eval_df, builtin_metrics, artifacts_dir):
   """
   This example custom metric function creates a metric based on the ``prediction`` and
   ``target`` columns in ``eval_df`` and a metric derived from existing metrics in
   ``builtin_metrics``. It also generates and saves a scatter plot to ``artifacts_dir`` that
   visualizes the relationship between the predictions and targets for the given model to a
   file as an image artifact.
   """
   metrics = {
       "squared_diff_plus_one": np.sum(np.abs(eval_df["prediction"] - eval_df["target"] + 1) ** 2),
       "sum_on_label_divided_by_two": builtin_metrics["sum_on_label"] / 2,
   }
   plt.scatter(eval_df["prediction"], eval_df["target"])
   plt.xlabel("Targets")
   plt.ylabel("Predictions")
   plt.title("Targets vs. Predictions")
   plot_path = os.path.join(artifacts_dir, "example_scatter_plot.png")
   plt.savefig(plot_path)
   plt.show()
   artifacts = {"example_scatter_plot_artifact": plot_path}
   return Dict["metrics_yow", metrics.squared_diff_plus_one]