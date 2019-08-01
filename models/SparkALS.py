# AWS EMR notebook code for calculating user and item factors.
# The code saves the output as a set of csv files

# Initiate Spark session
Spark

# Load necessary packages
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder, \
                                CrossValidatorModel
from pyspark.sql.functions import explode

# Import data
role = 'arn:aws:iam::260329411851:role/service-role/AmazonSageMaker-\
        ExecutionRole-20190722T132211'
bucket = 'fp-movielens-data'
region = 'us-east-1'
bucket_path = 'https://s3-{}.amazonaws.com/{}'.format(region, bucket)
filename = 's3://{}/{}'.format(bucket, 'ratings_processed.csv')
movie_ratings = spark.read.csv(filename, header='true', inferSchema='true')

# Build ALS model with grid search and cross-validation
# Examples of hyperparameter values for tuning are
# provided below. In practice, the parameters were
# tuned on a smaller version of the dataset (n=100,000).
# The choice to tune on the smaller dataset was made due to
# the computational and monetary costs associated with
# tuning on the complete dataset. 
als_model = ALS(userCol="userId", itemCol="movieId", ratingCol="rating",
                coldStartStrategy="drop")
               
params = ParamGridBuilder().addGrid(als_model.regParam, [0.1, 0.15, 0.3]) \
                            .addGrid(als_model.rank, [35, 42, 45]).build()
evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating",
                                predictionCol="prediction")

#  Instantiate crossvalidator estimator and fit model
cv = CrossValidator(estimator=als_model, estimatorParamMaps=params,
                    evaluator=evaluator, parallelism=4)
best_model = cv.fit(movie_ratings)

# Predict ratings and check RMSE
predictions = best_model.transform(movie_ratings)
rmse = evaluator.evaluate(predictions)
print("Root-mean-square error = " + str(rmse))

# Get parameters from the best model
rank = best_model.bestModel.rank
print("Rank: ", rank)
best_model.getEstimatorParamMaps()
best_model.avgMetrics

# Extract and save user factors
user_factors = best_model.bestModel.userFactors
user_factorsDF = (user_factors
                .select("id", explode("features")
                .alias("features"))
                .select('id', "features")
)
user_factorsDF.write.format("csv")\
    .save('s3://fp-movielens-data/user_factors.csv')

# Extract and save item factors
item_factors = best_model.bestModel.itemFactors
item_factorsDF = (item_factors
                    .select("id", explode("features")
                    .alias("features"))
                    .select('id', "features"))
item_factorsDF.write.format("csv").save('s3://fp-movielens-data/item_factors.csv')

#Kmeans

from pyspark.ml.feature import VectorAssembler
from pyspark.ml.clustering import KMeans
from numpy import array
from math import sqrt

role = 'arn:aws:iam::260329411851:role/service-role/AmazonSageMaker-ExecutionRole-20190722T132211'
bucket = 'fp-movielens-data'
region = 'us-east-1'
bucket_path = 'https://s3-{}.amazonaws.com/{}'.format(region, bucket)  # The URL to access the bucket
filename = 's3://{}/{}'.format(bucket, 'user_factors_scaled.csv')

user_factors = spark.read.csv(filename, header='true', inferSchema = 'true')

user_factors.head()

vecAssembler = VectorAssembler(inputCols=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 
                                         '10', '11', '12', '13', '14', '15', '16', '17', '18', '19',
                                         '20', '21', '22', '23', '24', '25', '26', '27', '28', '29',
                                         '30', '31', '32', '33', '34', '35', '36', '37', '38', '39',
                                         '40', '41'], outputCol="features")
new_df = vecAssembler.transform(user_factors)
new_df.show()

WSSSEs = []
K = range(2,60)
for k in K:
    kmeans = KMeans(k=k, seed=1)
    clusters = kmeans.fit(new_df.select('features'))
    wssse = clusters.computeCost(new_df.select('features'))
    WSSSEs.append(wssse)
    
kmeans = KMeans(k=18, seed=1)
clusters = kmeans.fit(new_df.select('features'))