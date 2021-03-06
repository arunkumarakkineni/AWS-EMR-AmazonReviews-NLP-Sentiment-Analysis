from pyspark.ml.classification import LogisticRegression
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.classification import NaiveBayes
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.mllib.evaluation import MulticlassMetrics

from pyspark.ml.evaluation import MulticlassClassificationEvaluator

class Model():

    def __init__(self,data,split,output_file_name,numClass):
        '''
        Model Class Constructor
        Input: data: model dataframe with feates column and the target label column, 
               split: train test split value between 0 to 1, 
               output_file_name: Name of the output file, numClass: Number of classes
        Output: None
        '''         
        #Generate the training and test datasets from the vectorized dataframe feature column
        self.data = data
        self.filename = output_file_name
        self.trainingData, self.testData = self.data.randomSplit([split, 1-split],seed=100123)
 
        if numClass == 3:          
            #Count the number of positive, neutral, and negative reviews
            self.trainingDataCount = self.trainingData.count()
            self.testDataCount = self.testData.count()
            self.testPosReviewsCount = self.testData.filter(self.testData.label == 2).count()
            self.testNeuReviewsCount = self.testData.filter(self.testData.label == 1).count()
            self.testNegReviewsCount = self.testData.filter(self.testData.label == 0).count()
            self.trainPosReviewsCount = self.trainingData.filter(self.trainingData.label == 2).count()
            self.trainNeuReviewsCount = self.trainingData.filter(self.testData.label == 1).count()
            self.trainNegReviewsCount = self.trainingData.filter(self.testData.label == 0).count()
        elif numClass == 2:
            self.trainingDataCount = self.trainingData.count()
            self.testDataCount = self.testData.count()
            self.testPosReviewsCount = self.testData.filter(self.testData.label == 1).count()
            self.testNegNeuReviewsCount = self.testData.filter(self.testData.label == 0).count()
            self.trainPosReviewsCount = self.trainingData.filter(self.trainingData.label == 1).count()
            self.trainNegNeuReviewsCount = self.trainingData.filter(self.testData.label == 0).count()
            
         
        #Setup the evaluation metrics
        self.evaluator = MulticlassClassificationEvaluator(predictionCol="prediction")

    def model_evaluator(self, predictions, modelType, modelParams, numClass):
        '''
        Evaluates the model and stores the results in text file initialized by the model class constructor 
        Input: dataframe with model predictions (predictions), modelType: {randomforest,logistic regression, naive bayes, gradient boost},
               modelParams: {Parameter list for each model type}, numClass: {[pos,neg,neutral],[pos, neg_neu]}
        Output: None
        ''' 
        #Evaluate the results generated by the model
        evaluator = self.evaluator.evaluate(predictions)

        if numClass == 3:
            predictions.show(10)
            predictions.printSchema()
            precision_pos_predictions = predictions.filter((predictions['prediction']==2) & (predictions['label']==2)).count()/self.testPosReviewsCount
            precision_neu_predictions = predictions.filter((predictions['prediction']==1) & (predictions['label']==1)).count()/self.testNeuReviewsCount
            precision_neg_predictions = predictions.filter((predictions['prediction']==0) & (predictions['label']==0)).count()/self.testNegReviewsCount

            with open(self.filename,'a') as f:
                f.write("\n\n\n"+modelType+": "+" NumClass=3 " +modelParams+"\n\n"+"Summary Stats: \n\n")
                f.write("Overall Accuracy = {0}, Positive Class Precision = {1}, Neutral Class Precision = {2}, Negative Class Precision = {3}, Num Positive Reviews in testdata = {4}, Num Neutral Reviews in testdata = {5}, Num Negative Reviews in testdata = {6},Num Positive Reviews in traindata = {7}, Num Neutral Reviews in traindata = {8}, Num Negative Reviews in traindata = {9} ".format(evaluator,precision_pos_predictions,precision_neu_predictions,precision_neg_predictions,self.testPosReviewsCount,self.testNeuReviewsCount,self.testNegReviewsCount,self.trainPosReviewsCount,self.trainNeuReviewsCount,self.trainNegReviewsCount)) 
        elif numClass == 2:
            predictions.show(10)
            predictions.printSchema()
            precision_pos_predictions = predictions.filter((predictions['prediction']==1) & (predictions['label']==1)).count()/self.testPosReviewsCount
            precision_negneu_predictions = predictions.filter((predictions['prediction']==0) & (predictions['label']==0)).count()/self.testNegNeuReviewsCount

            with open(self.filename,'a') as f:
                f.write("\n\n\n"+modelType+": "+" NumClass=2 "+modelParams+"\n\n"+"Summary Stats: \n\n")
                f.write("Overall Accuracy = {0}, Positive Class Precision = {1}, Neg Neu Class Precision = {2}, Num Positive Reviews in testdata = {3}, Num Neg Neu Reviews in testdata = {4}, Num Positive Reviews in traindata = {5}, Num Neg Neu Reviews in traindata = {6}".format(evaluator,precision_pos_predictions,precision_negneu_predictions,self.testPosReviewsCount,self.testNegNeuReviewsCount,self.trainPosReviewsCount,self.trainNegNeuReviewsCount))

        predictions.select('label','prediction').show(1000)

    def exec_random_forest(self,featuresCol1="features", labelCol1="label", predictionCol1="prediction", 
                           numTrees1=20, maxDepth1=5, maxBins1=32, numClass1=2):
        '''
        Creates the RandomForest model Pipeline
        Input: featureCol1: feature column name, labelCol: label column name, predictionCol1: prediction column name
                            model parameters: {numTrees, maxDepth, maxBins}, numClass1: number of class labels
        Output: None
        ''' 
        #Initialize RandomForest Model with parameters passed
        rf = RandomForestClassifier(labelCol = labelCol1, featuresCol = featuresCol1, predictionCol = predictionCol1,
                                    numTrees = numTrees1, maxDepth = maxDepth1, 
                                    maxBins = maxBins1)

        #Fit RF model with training data
        rfModel = rf.fit(self.trainingData)

        #Make RF model predictions on testData
        predictions = rfModel.transform(self.testData)

        #Evaluate the results generated by the model prediction 
        self.model_evaluator(predictions,
                             modelType="RandomForest Model",
                             modelParams = str({'numTrees':numTrees1,'maxDepth':maxDepth1,'maxBins':maxBins1}),
                             numClass=numClass1)


    def exec_naive_bayes(self,featuresCol1="features", labelCol1="label", predictionCol1="prediction",
                         smoothing1 = 1, numClass1 = 2):
        '''
        Creates the Naive Bayes model Pipeline
        Input: featureCol1: feature column name, labelCol: label column name, predictionCol1: prediction column name
                            model parameters: {smoothing}, numClass1: number of class labels
        Output: None
        ''' 
        #Initialize NaiveBayes Model with parameters passed
        nb = NaiveBayes(featuresCol=featuresCol1,
                        labelCol=labelCol1,predictionCol = predictionCol1,
                        smoothing=smoothing1)

        #Fit nb model with training data
        nbModel = nb.fit(self.trainingData)

        #Make nb model predictions on testData
        predictions = nbModel.transform(self.testData)

        #Evaluate the results generated by the model prediction 
        self.model_evaluator(predictions,
                             modelType="NaiveBayes Model",
                             modelParams = str({'smoothing':smoothing1}),
                             numClass = numClass1)

    def exec_logistic_regression(self,featuresCol1="features", labelCol1="label", predictionCol1="prediction",
                                 maxIter1=30,regParam1=0.3,elasticNetParam1=0, numClass1=2):
        '''
        Creates the Logistic Regression model Pipeline
        Input: featureCol1: feature column name, labelCol: label column name, predictionCol1: prediction column name
                            model parameters: {max iterations, regularization parameter, elastic net parameter}, 
                            numClass1: number of class labels
        Output: None
        ''' 
        #Initialize Logistic Regression Model with parameters passed
        lr = LogisticRegression(featuresCol=featuresCol1,
                                labelCol=labelCol1,predictionCol = predictionCol1,
                                maxIter=maxIter1,regParam=regParam1,elasticNetParam=elasticNetParam1)

        #Fit lr model with training data
        lrModel = lr.fit(self.trainingData)

        #Make lr model predictions on testData
        predictions = lrModel.transform(self.testData)

        #Evaluate the results generated by the model prediction 
        self.model_evaluator(predictions,
                             modelType="Logistic Regression Model",
                             modelParams = str({'maxIter':maxIter1,'regParam':regParam1,'elasticNetParam':elasticNetParam1}),
                             numClass = numClass1)
        
    def exec_gradient_boost(self,featuresCol1="features", labelCol1="label", predictionCol1="prediction",
                            maxIter1 = 30, numClass1=2):
        '''
        Creates the Gradient Boost model Pipeline, this model is only applicable to binary classification problems
        Input: featureCol1: feature column name, labelCol: label column name, predictionCol1: prediction column name
                            model parameters: {maximum number of iterations}, numClass1: number of class labels restricted to 2
        Output: None
        ''' 
        #Initialize GradientBoost Model with parameters passed
        gb = GBTClassifier(featuresCol=featuresCol1,
                           labelCol=labelCol1,predictionCol = predictionCol1,
                           maxIter=maxIter1)

        #Fit gradient boost model with training data
        gbModel = gb.fit(self.trainingData)

        #Make nb model predictions on testData
        predictions = gbModel.transform(self.testData)

        #Evaluate the results generated by the model prediction 
        self.model_evaluator(predictions,
                             modelType="GradientBoost Model",
                             modelParams = str({'maxIter':maxIter1}),
                             numClass=numClass1)
