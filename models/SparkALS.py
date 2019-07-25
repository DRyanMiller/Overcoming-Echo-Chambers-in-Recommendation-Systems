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