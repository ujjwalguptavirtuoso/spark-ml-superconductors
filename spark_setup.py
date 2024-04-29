from pyspark.sql import SparkSession
from pyspark.streaming import StreamingContext

def create_spark_session(app_name="Superconductor Temperature Prediction"):
    spark = SparkSession.builder \
        .master("local[*]") \
        .appName(app_name) \
        .config("spark.driver.memory", "4g") \
        .config("spark.executor.memory", "4g")  \
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
        .config("spark.shuffle.compress", "true") \
        .config("spark.shuffle.spill.compress", "true") \
        .getOrCreate()
    return spark

def create_streaming_context(spark, batch_interval=1):
    ssc = StreamingContext(spark.sparkContext, batch_interval)
    return ssc
