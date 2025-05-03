import os
import sys
import argparse
import configparser
from logger import Logger
from functools import reduce
from dotenv import load_dotenv

sys.path.append(os.environ['SPARK_HOME'] + '/python')
sys.path.append(os.environ['SPARK_HOME']+ '/python/build')
sys.path.append(os.environ['SPARK_HOME'] + '/python/pyspark')
sys.path.append(os.environ['SPARK_HOME'] + '/python/lib/py4j-0.10.9.7-src.zip')

from pyspark.sql import SparkSession
from pyspark.conf import SparkConf
from pyspark.sql.functions import col
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.sql.types import FloatType

SHOW_LOG = True


class Clusterizer():
    def __init__(self):
        logger = Logger(SHOW_LOG)
        self.config = configparser.ConfigParser()
        self.log = logger.get_logger(__name__)
        
        load_dotenv()
        
        self.csv_path = "./sparkdata/en.openfoodfacts.org.products.csv"
        spark_config_apth = 'conf/spark.ini'
        self.useful_cols = [
            'code',
            'energy-kcal_100g',
            'fat_100g',
            'carbohydrates_100g',
            'sugars_100g',
            'proteins_100g',
            'salt_100g',
            'sodium_100g',
        ]
        self.metadata_cols = ['code']
        self.feature_cols = [c for c in self.useful_cols if c not in self.metadata_cols]
        
        self.config = configparser.ConfigParser()
        self.config.optionxform = str
        self.config.read(spark_config_apth)
        
        conf = SparkConf()
        config_params = list(self.config['spark'].items())
        conf.setAll(config_params)
        
        params_str = '\n'.join([f'{k}: {v}' for k, v in config_params])
        self.log.info(f"Spark App Configuration Params:\n{params_str}")
            
        self.spark = SparkSession.builder.config(conf=conf) \
            .getOrCreate()
                        
    def prepare_df(self, df):
        processed_df = df.select(*self.useful_cols).na.drop()

        for column in self.feature_cols:
            processed_df = processed_df.withColumn(column, col(column).cast(FloatType()))
        
        # an energy-amount of more than 1000kcal 
        # (the maximum amount of energy a product can have; 
        # in this case it would conists of 100% fat)
        processed_df = processed_df.filter(col('energy-kcal_100g') < 1000)
        
        # a feature (except for the energy-ones) higher than 100g
        columns_to_filter = [c for c in processed_df.columns if c != 'energy-kcal_100g' and c not in self.metadata_cols]
        condition = reduce(
            lambda a, b: a & (col(b) < 100),
            columns_to_filter,
            col(columns_to_filter[0]) < 100 
        )
        processed_df = processed_df.filter(condition)
        
        # a feature with a negative entry
        condition = reduce(
            lambda a, b: a & (col(b) >= 0),
            self.feature_cols,
            col(self.feature_cols[0]) >= 0 
        )
        processed_df = processed_df.filter(condition)
        
        return processed_df
        
    def cluster(self, cluster_df):
        cluster_count = int(self.config['model']['k'])
        seed = int(self.config['model']['seed'])

        kmeans = KMeans(k=cluster_count).setSeed(seed)
        kmeans_model = kmeans.fit(cluster_df)
        
        return kmeans_model
    
    def evaluate(self, cluster_df):
        evaluator = ClusteringEvaluator(metricName="silhouette", distanceMeasure="squaredEuclidean")
        score = evaluator.evaluate(cluster_df)
        return score
    
    def save_results(self, init_df, transformed):
        init_df.join(transformed.select(*metadata_cols, "prediction"), on="code", how="left") \
            .select(*metadata_cols, 'prediction') \
            .write.format("csv") \
            .mode("overwrite") \
            .save("./sparkdata/predictions.csv")
    
    def run(self):
        df = self.spark.read.option("delimiter", "\t") \
            .csv(self.csv_path, header=True, inferSchema=True) \
            .cache()
            
        processed_df = self.prepare_df(df)   
        self.log.info(f"{processed_df.count()} lines left after preprocessing")
        
        cluster_df = VectorAssembler(
            inputCols=self.feature_cols, 
            outputCol="features"
        ).transform(processed_df)
        
        model = self.cluster(cluster_df)
        self.log.info(f"Class distribution: {model.summary.clusterSizes}")
        
        pred_df = model.transform(cluster_df)
        
        score = self.evaluate(pred_df)
        self.log.info(f"Silhouette Score: {score}")
         
        self.save_results(df, pred_df)
        self.log.info('Results saved successfully!')
        
        self.log.info('Stopping spark app...')
        self.spark.stop()


if __name__ == "__main__":
    clusterizer = Clusterizer()
    clusterizer.run()
