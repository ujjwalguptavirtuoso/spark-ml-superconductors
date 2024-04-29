# Import the necessary modules
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegressionModel
from spark_setup import create_spark_session  # Import from your existing setup file


def predict_new_data(model_path):
    # Use the imported function to create a Spark session
    spark = create_spark_session("Model Prediction")

    # Load the pre-trained model from the specified path
    model = LinearRegressionModel.load(model_path)

    # Define in-memory sample data to predict
    data = [(Vectors.dense([1.0,100.5,200.5]),), (Vectors.dense([2.0,110.5,210.5]),)]
    columns = ["features"]
    new_data = spark.createDataFrame(data, columns)

    # Use the model to make predictions
    predictions = model.transform(new_data)

    # Display the predictions
    predictions.show()


if __name__ == "__main__":
    model_path = "/Users/ujjwalgupta/Documents/ml_models/"
    predict_new_data(model_path)
