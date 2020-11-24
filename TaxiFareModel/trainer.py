import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from TaxiFareModel.encoders import TimeFeaturesEncoder
from TaxiFareModel.encoders import DistanceTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from TaxiFareModel.utils import haversine_vectorized
from TaxiFareModel.utils import extract_time_features
from TaxiFareModel.data import get_data
from TaxiFareModel.data import clean_data
from TaxiFareModel.utils import compute_rmse

class Trainer():
    def __init__(self, X, y):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.X = X
        self.y = y
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2)

    def set_pipeline(self):
        distpipe = Pipeline([
            ('imputer', DistanceTransformer()),
            ('scaler', StandardScaler())
        ])
        timepipe = Pipeline([
            ('imputer', TimeFeaturesEncoder('pickup_datetime')),
            ('encoder', OneHotEncoder(handle_unknown = 'ignore'))
        ])
        preprocessor = ColumnTransformer([
            ('dist_transformer', distpipe, ['pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude']),
            ('time_transformer', timepipe, ['pickup_datetime'])])
        self.pipeline = Pipeline([
            ('preprocessing', preprocessor),
            ('linear_regression', LinearRegression())])

    def run(self):
        """set and train the pipeline"""
        self.pipeline.fit(self.X_train,self.y_train)


    def evaluate(self):
        return compute_rmse(self.pipeline.predict(self.X_test),self.y_test)

    @memoized_property
    def mlflow_client(self):
        mlflow.set_tracking_uri(MLFLOW_URI)
        return MlflowClient()

    @memoized_property
    def mlflow_experiment_id(self):
        try:
            return self.mlflow_client.create_experiment(self.experiment_name)
        except BaseException:
            return self.mlflow_client.get_experiment_by_name(self.experiment_name).experiment_id

    @memoized_property
    def mlflow_run(self):
        return self.mlflow_client.create_run(self.mlflow_experiment_id)

    def mlflow_log_param(self, key, value):
        self.mlflow_client.log_param(self.mlflow_run.info.run_id, key, value)

    def mlflow_log_metric(self, key, value):
        self.mlflow_client.log_metric(self.mlflow_run.info.run_id, key, value)


if __name__ == "__main__":
    df = get_data(nrows=10000)
    X = df.drop(columns = 'fare_amount')
    y = df['fare_amount']
    pipeline = Trainer(X,y)
    pipeline.set_pipeline()
    pipeline.run()
    result = pipeline.evaluate()
    print(result)
