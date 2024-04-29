# train_and_save_model.py
from pyspark.sql.types import StructType, StructField, FloatType, IntegerType
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from spark_setup import create_spark_session, create_streaming_context
from schema_definition import get_superconductors_schema

# Define the schema based on your dataset
# schema = StructType([
#     StructField("number_of_elements", IntegerType()),
#     StructField("mean_atomic_mass", FloatType()),
#     StructField("wtd_mean_atomic_mass", FloatType()),
#     StructField("critical_temp", FloatType())
# ])

schema = get_superconductors_schema()

def parse_line(line):
    print("Parsing line:", line)
    parts = line.split(',')
    return tuple(float(x) for x in parts)

def process_rdd(rdd):
    print("Processing RDD, isEmpty:", rdd.isEmpty()) # Check if RDDs are empty
    if not rdd.isEmpty():
        df = spark.createDataFrame(rdd, schema=schema)
        assembler = VectorAssembler(inputCols=df.columns[:-1], outputCol="features")
        transformed_df = assembler.transform(df).select("features", "critical_temp")
        print("Processed DataFrame:", transformed_df.take(5))  # Output some processed data
        return transformed_df.rdd
    return rdd

def train_and_save_model(rdd, model_path):
    if not rdd.isEmpty():
        df = spark.createDataFrame(rdd, schema=["features", "critical_temp"])
        lr = LinearRegression(featuresCol="features", labelCol="critical_temp")
        model = lr.fit(df)
        model.write().overwrite().save(model_path)

if __name__ == "__main__":
    spark = create_spark_session()
    ssc = create_streaming_context(spark, batch_interval=5)  # Adjust batch interval as needed

    input_directory = "/Users/ujjwalgupta/Documents/input_data/"
    model_path = "/Users/ujjwalgupta/Documents/ml_models/"

    data_stream = ssc.textFileStream(input_directory)
    transformed_data = data_stream.map(parse_line)
    processed_data = transformed_data.transform(process_rdd)
    processed_data.foreachRDD(lambda rdd: train_and_save_model(rdd, model_path))

    ssc.start()
    ssc.awaitTermination()
