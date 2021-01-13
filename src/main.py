import pyspark as ps
import numpy as np

from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql import SQLContext

import pipeline
import model

if __name__ == '__main__':

    spark = (
            ps.sql.SparkSession.builder
            .appName("aka")
            .getOrCreate()
            )

    sc = spark.sparkContext

    # Read in data from S3
    home_and_tools = 's3://s3-aka-emr-cluster/data/Tools_and_Home_Improvement_large.json'
    home_and_tools_local = 'file:///home/hadoop/Tools_and_Home_Improvement_5.json'
    df = spark.read.json(home_and_tools)
    df.count()
    df1 = df.select('overall','reviewText').filter((df['overall'] == 1.0) | (df['overall'] == 2.0) | (df['overall'] == 3.0) | (df['overall'] == 4.0) | (df['overall'] == 5.0)).distinct().dropna()
    df1.show(10)

    ###############################
    # Model scores for TFIDF only.
    ###############################

    p = pipeline.DataPipeline(df1,spark,sc)
    #Create Dataframe with 8000000 rows
    p.select_data_multiclass(8000000)
    #p.select_data_binaryclass(1000000)
    #p.select_data_binaryclass(1500000)

    df_vec = p.add_vectorized_features(transform_type='tfidf',
                                       min_df = 5,
                                       max_df = 0.5,
                                       isCHISQR = False,
                                       chi_feature_num = 500,
                                       num_features = 20000)
    df_vec.show(30)
    df_vec.printSchema()
    df_vec.select('features','label').show(100)
     
    m = model.Model(df_vec,0.7,'TFIDF_vectorization_results.txt',3)
    #m.exec_random_forest("features","label","prediction",200,30,32,3)
    #m.exec_naive_bayes("features","label","prediction",1,3)
    m.exec_logistic_regression("features", "label","prediction",40,0.3,0.0,3)
    #m.exec_gradient_boost("features","label","prediction",200,3)

    #Uncomment the code below as needed to build the pySpark feature pipeline 
    '''
    ############################################################
    #Run TFIDF + isCHISQR = TRUE, class = binary (pos, neg_neu)
    ###########################################################
    p1 = pipeline.DataPipeline(df1,spark,sc)
    p1.select_data_binaryclass(1500000)
    df_vec1 = p1.add_vectorized_features(transform_type='tfidf',
                                         min_df = 5,
                                         max_df = 0.5,
                                         isCHISQR = True,
                                         chi_feature_num = 1000,
                                         num_features = 20000)
    df_vec1.show(30)
    df_vec1.printSchema()
    df_vec1.select('features','label').show(100)
    #m1 = model.Model(df_vec1,0.7,'TFIDF_vectorization_chisquare_results.txt',2)
    #m1.exec_random_forest("features","label","prediction",200,30,32,2)
    #m1.exec_naive_bayes("features","label","prediction",1,2)
    #m1.exec_logistic_regression("features", "label","prediction",40,0.3,0,2)
    #m1.exec_gradient_boost("features","label","prediction",200,2)
    '''
    '''
    ###############################################################
    #Run TFIDF + isCHISQR = TRUE, class = multiclass (neg,neu,pos)
    ##############################################################
    p2 = pipeline.DataPipeline(df1,spark,sc)
    p2.select_data_multiclass(1500000)
    df_vec2 = p2.add_vectorized_features(transform_type='tfidf',
                                         min_df = 5,
                                         max_df = 0.5,
                                         isCHISQR = True,
                                         chi_feature_num = 1000,
                                         num_features = 20000)
    df_vec2.show(30)
    df_vec2.printSchema()
    df_vec2.select('features','label').show(100)
    #m2 = model.Model(df_vec2,0.7,'TFIDF_vectorization_chisquare_results.txt',3)
    #m2.exec_random_forest("features","label","prediction",200,30,32,3)
    #m2.exec_naive_bayes("features","label","prediction",1,3)
    #m2.exec_logistic_regression("features", "label","prediction",40,0.3,0,3)
    '''
    '''
    #############################################################
    #Run TFIDF + isCHISQR = FALSE, class = binary (pos, neg_neu)
    ############################################################
    p3 = pipeline.DataPipeline(df1,spark,sc)
    p3.select_data_binaryclass(1500000)
    df_vec3 = p3.add_vectorized_features(transform_type='tfidf',
                                         min_df = 5,
                                         max_df = 0.5,
                                         isCHISQR = False,
                                         chi_feature_num = 1000,
                                         num_features = 20000)
    df_vec3.show(30)
    df_vec3.printSchema()
    df_vec3.select('features','label').show(100)
    #m3 = model.Model(df_vec3,0.7,'TFIDF_vectorization_chisquare_results.txt',2)
    #m3.exec_random_forest("features","label","prediction",200,30,32,2)
    #m3.exec_naive_bayes("features","label","prediction",1,2)
    #m3.exec_logistic_regression("features", "label","prediction",40,0.3,0,2)
    #m3.exec_gradient_boost("features","label","prediction",200,2)
    '''
    '''
    ############################################################
    #Run NGRAM + isCHISQR = TRUE, class = binary (pos, neg_neu)
    ###########################################################
    p4 = pipeline.DataPipeline(df1,spark,sc)
    p4.select_data_binaryclass(1500000)
    df_vec4 = p4.add_vectorized_features(transform_type='tfidf_bigram',
                                         min_df = 5,
                                         max_df = 0.5,
                                         isCHISQR = True,
                                         chi_feature_num = 1000,
                                         num_features = 20000)
    df_vec4.show(30)
    df_vec4.printSchema()
    df_vec4.select('features','label').show(100)
    #m4 = model.Model(df_vec4,0.7,'NGRAM_vectorization_chisquare_results.txt',2)
    #m4.exec_random_forest("features","label","prediction",200,30,32,2)
    #m4.exec_naive_bayes("features","label","prediction",1,2)
    #m4.exec_logistic_regression("features", "label","prediction",40,0.3,0,2)
    #m4.exec_gradient_boost("features","label","prediction",200,2)
    '''
    '''
    ############################################################
    #Run NGRAM + isCHISQR = FALSE, class = binary (pos, neg_neu)
    ###########################################################
    p5 = pipeline.DataPipeline(df1,spark,sc)
    p5.select_data_binaryclass(1500000)
    df_vec5 = p5.add_vectorized_features(transform_type='tfidf_bigram',
                                         min_df = 5,
                                         max_df = 0.5,
                                         isCHISQR = False,
                                         chi_feature_num = 1000,
                                         num_features = 20000)
    df_vec5.show(30)
    df_vec5.printSchema()
    df_vec5.select('features','label').show(100)
    #m5 = model.Model(df_vec5,0.7,'NGRAM_vectorization_results.txt',2)
    #m5.exec_random_forest("features","label","prediction",200,30,32,2)
    #m5.exec_naive_bayes("features","label","prediction",1,2)
    #m5.exec_logistic_regression("features", "label","prediction",40,0.3,0,2)
    #m5.exec_gradient_boost("features","label","prediction",200,2)
    '''
    '''
    ############################################################
    #Run NGRAM + isCHISQR = TRUE, class = multiclass (pos, neg, neu)
    ###########################################################
    p6 = pipeline.DataPipeline(df1,spark,sc)
    p6.select_data_multiclass(1500000)
    df_vec6 = p6.add_vectorized_features(transform_type='tfidf_bigram',
                                         min_df = 5,
                                         max_df = 0.5,
                                         isCHISQR = True,
                                         chi_feature_num = 1000,
                                         num_features = 20000)
    df_vec6.show(30)
    df_vec6.printSchema()
    df_vec6.select('features','label').show(100)
    #m6 = model.Model(df_vec6,0.7,'NGRAM_vectorization_chisquare_results.txt',3)
    #m6.exec_random_forest("features","label","prediction",200,30,32,3)
    #m6.exec_naive_bayes("features","label","prediction",1,3)
    #m6.exec_logistic_regression("features", "label","prediction",40,0.3,0,3)
    '''
    '''
    ############################################################
    #Run NGRAM + isCHISQR = FALSE, class = multiclass (pos, neg, neu)
    ###########################################################
    p7 = pipeline.DataPipeline(df1,spark,sc)
    p7.select_data_multiclass(1500000)
    df_vec7 = p7.add_vectorized_features(transform_type='tfidf_bigram',
                                         min_df = 5,
                                         max_df = 0.5,
                                         isCHISQR = False,
                                         chi_feature_num = 1000,
                                         num_features = 20000)
    df_vec7.show(30)
    df_vec7.printSchema()
    df_vec7.select('features','label').show(100)
    #m7 = model.Model(df_vec7,0.7,'NGRAM_vectorization_results.txt',3)
    #m7.exec_random_forest("features","label","prediction",200,30,32,3)
    #m7.exec_naive_bayes("features","label","prediction",1,3)
    #m7.exec_logistic_regression("features", "label","prediction",40,0.3,0,3)
    ''' 
