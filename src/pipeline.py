import nltk
from pyspark.sql.types import DoubleType
from pyspark.sql.functions import lit, udf, coalesce
from pyspark.sql import SQLContext
from pyspark.ml import Pipeline
from pyspark.ml.feature import (RegexTokenizer,
                                StopWordsRemover,
                                CountVectorizer,
                                OneHotEncoder,
                                StringIndexer,
                                VectorAssembler,
                                NGram,
                                Word2Vec,
                                StandardScaler,
                                HashingTF,
                                IDF,
                                PCA,
                                ChiSqSelector)

class DataPipeline():

    def __init__(self, pyspark_df, spark_session, sc):
        self.df = pyspark_df
        self.session = spark_session
        self.sc = sc
    
    def select_data_multiclass(self,num_rows):
        '''
        Creates the dataset for the pySpark pipeline and three categories pos,neg,neu in a label column for train/test targets
        Input: Number of rows to be selected from the cleaned data for model development
        Output: None
        '''        
        self.df.registerTempTable('reviewsTable')
        # Spark dataframe that will get vectorized and put in a model.
        # Create label column that will be used as labels for training a tri-class classifier
        self.df = self.session.sql("""select
                                      overall,
                                      reviewText,
                                      case when overall >= 4.0 then double(2)
                                           when overall == 3.0 then double(1)
                                           when overall < 3.0 then  double(0)
                                      end as label
                                      from reviewsTable
                                      limit {limit_val}
                                    """.format(limit_val = num_rows))
        
        '''
        df_neg_neu = self.df.filter((self.df['label'] == 0) | (self.df['label'] == 1))
        df_pos = self.df.filter(self.df['label'] == 2).limit(int(df_neg_neu.count()/2))
        self.df = df_pos.union(df_neg_neu)
        '''
        df_neu = self.df.filter(self.df['label'] == 1)
        df_pos = self.df.filter(self.df['label'] == 2).limit(int(df_neu.count()))
        df_neg = self.df.filter(self.df['label'] == 0).limit(int(df_neu.count()))
        self.df = df_neg.union(df_pos.union(df_neu))        
        print("pos = {0}, neu= {1}, neg = {2}".format(self.df.filter(self.df['label'] == 2).count(),self.df.filter(self.df['label'] == 1).count(),self.df.filter(self.df['label'] == 0).count()))
        self.df.show(100)
        self.df.printSchema()        

    def select_data_binaryclass(self,num_rows):
        '''
        Creates the dataset for the pySpark pipeline and two categories pos,neg_neu in a label column for train/test targets
        Input: Number of rows to be selected from the cleaned data for model development
        Output: None
        ''' 
        self.df.registerTempTable('reviewsTable')
        # Spark dataframe that will get vectorized and put in a model.
        # Create label column that will be used as labels for training a tri-class classifier
        self.df = self.session.sql("""select
                                      overall,
                                      reviewText,
                                      case when overall >= 4.0 then double(1)
                                           when overall == 3.0 then double(0)
                                           when overall < 3.0 then  double(0)
                                      end as label
                                      from reviewsTable
                                      limit {limit_val}
                                    """.format(limit_val = num_rows))
        df_neg_neu = self.df.filter((self.df['label'] == 0))
        df_pos = self.df.filter(self.df['label'] == 1).limit(int(df_neg_neu.count()))
        self.df = df_pos.union(df_neg_neu)
        print("pos = {0}, neu neg = {1}".format(self.df.filter(self.df['label'] == 1).count(),self.df.filter(self.df['label'] == 0).count()))
        self.df.show(1000)
        self.df.printSchema()

    def add_vectorized_features(self,transform_type,min_df,max_df,
                                isCHISQR,chi_feature_num,num_features):
        '''
        Creates the pySpark feature pipeline and stores the vectorized data under the feature column 
        Input: transform_type: {'tfidf','tfidf_bigram'}, min document frequency (min_df), chi squared feature reduction (isCHISQR)
               number of reduced features with chi square feature reduction (chi_feature_num), number of features (num_features)                  
        Output: Returns the transformed dataframe with the label and features columns
        ''' 
        stages = []
        #Code this code transforms text to vectorized features
         
        # Tokenize review sentences into vectors of words
        regexTokenizer = RegexTokenizer(inputCol="reviewText",
                                        outputCol="words",
                                        pattern="\\W")
        
        stages+=[regexTokenizer]
        
        #Remove stopwords from tokenized words
        #nltk.download('stopwords')
        from nltk.corpus import stopwords
        sw = stopwords.words('english')
        stopwordsRemover = StopWordsRemover(inputCol="words",
                                            outputCol="filtered").setStopWords(sw)

        #lemmatizer = WordNetLemmatizer()
        #doc = [lemmatizer.lemmatize(token) for token in doc]
        stages+=[stopwordsRemover]
 
        # Using TFIDF for review transformation of unigrams.
        if transform_type == 'tfidf':
            # Creating IDF from the words the filtered words
            hashingTF = HashingTF(inputCol="filtered",
                                  outputCol="rawFeatures",
                                  numFeatures=num_features)
            idf = IDF(inputCol="rawFeatures",
                      outputCol="review_vector",
                      minDocFreq=min_df)
            # Add to stages
            stages += [hashingTF,idf]
        
        # Using TFIDF for review transformation of bigrams
        if transform_type == 'tfidf_bigram':
            #Add unigram and bigram word vectors, then vectorize using TFIDF
            unigram = NGram(n=1,inputCol='filtered',outputCol='unigrams')
            stages+=[unigram]
            
            bigram = NGram(n=2,inputCol='filtered',outputCol='bigrams')
            stages+=[bigram]
            # Creating IDF from unigram  words
            hashingTF_unigram = HashingTF(inputCol="unigrams",
                                          outputCol="rawFeatures_unigrams",
                                          numFeatures=num_features)
            idf_unigram = IDF(inputCol="rawFeatures_unigrams",
                              outputCol="unigrams_vector",
                              minDocFreq=min_df)
            # Add to stages
            stages += [hashingTF_unigram,idf_unigram]
            # Creating IDF from the bigram words
            hashingTF_bigram = HashingTF(inputCol="bigrams",
                                          outputCol="rawFeatures_bigrams",
                                          numFeatures=num_features)
            idf_bigram = IDF(inputCol="rawFeatures_bigrams",
                              outputCol="bigrams_vector",
                              minDocFreq=min_df)
            # Add to stages
            stages += [hashingTF_bigram,idf_bigram]
            
            ngrams = VectorAssembler(inputCols=['unigrams_vector','bigrams_vector'],
                                     outputCol='review_vector')
            stages += [ngrams]

        assemblerInputs = ['review_vector']
        assembler = VectorAssembler(inputCols=assemblerInputs,
                                    outputCol="unstandard_features")
        
        stages += [assembler]

        if isCHISQR:
            chi_selector = ChiSqSelector(numTopFeatures=chi_feature_num,
                                         featuresCol="unstandard_features",
                                         outputCol="chisq_features",
                                         labelCol="label")

            stages += [chi_selector]

            scaler = StandardScaler(inputCol="chisq_features",
                                    outputCol="features",
                                    withStd=True,
                                    withMean=False)

            stages += [scaler]
        else:
            scaler = StandardScaler(inputCol="unstandard_features",
                                    outputCol="features",
                                    withStd=True,
                                    withMean=False)

            stages += [scaler]    
        
        pipeline = Pipeline(stages=stages)
        pipelineFit = pipeline.fit(self.df)
        self.df = pipelineFit.transform(self.df)
        return self.df
