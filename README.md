# Addressing the Echo Chamber Problem in Recommendation Systems Using Movie Ratings
==============================

The purpose of this project is to create a recommendation system that overcomes the echo chamber problem.  The echo chamber problem occurs when additional information provided to a recommendation system results in a decrease in the diversity of recommendations for a given user.  The result is a recommendation system that reinforces a user’s recorded preferences and fails to provide recommendations outside the user’s typical purview. 

In order to overcome the echo chamber problem, this project builds a system that provides additional recommendations based on the preferences of others who are different, but not too different, from the user.  These additional recommendations are identified by clustering users based on the latent user features from an ALS decomposition of a ratings matrix. Once clusters are created, the top-rated items from users in each cluster can be identified. The top-rated items in each cluster form the set of items from which the recommendations will be extracted.  To get recommendations for a user, the following steps occur:
1.	The distance between clusters is measured,
2.	A user is classified into a cluster,
3.	From the nearest cluster to the user’s cluster, a random selection of items is recommended to the user.
4.	Step three can be repeated for the k nearest clusters.  In such a case, the proportion of total recommendations selected from a cluster is weighted by the cluster’s distance from the user’s cluster. (For example, for k=3 and a total of 10 recommendations, 5 recommendations may be selected from the nearest cluster, 3 from the second nearest, and 2 from the third nearest.)

The project utilizes the [MovieLens](https://grouplens.org/datasets/movielens/) dataset (Harper & Konstan, 2015). The MovieLens dataset is an open source data set containing 27,753,444 movie ratings from 283,228 users for 58,098 movies. The ratings are on a five-star scale range from 0.5 stars to 5 stars in 0.5 star increments. The files include data from January 09, 1995 and September 26, 2018. The data set includes a random sample of users with at least 1 movie rating.

==============================

Project Organization
------------
The directory structure for this projects is below. Brief descriptions follow the diagram.

```
Echo_Chamber
├── LICENSE
│
├── Makefile  : Makefile with commands to perform selected tasks, e.g. `make clean_data`
│
├── README.md : Project README file containing description provided for project.
│
├── .env      : file to hold environment variables (see below for instructions)
│
├── test_environment.py
│
├── data
│   ├── processed : directory to hold interim and final versions of processed data.
│   └── raw : holds original data. This data should be read only.
│
├── models  : holds binary, json (among other formats) of initial, trained models.
│
├── notebooks : holds notebooks for eda, model, and evaluation. Use naming convention yourInitials_projectName_useCase.ipynb
│
├── references : data dictionaries, user notes project, external references.
│
├── reports : interim and final reports, slides for presentations, accompanying visualizations.
│   └── figures
│
├── requirements.txt
│
├── setup.py
│
├── src : local python scripts. Pre-supplied code stubs include clean_data, model and visualize.
    ├── __make_data__.py
    ├── __settings__.py
    ├── clean_data.py
    ├── custom.py
    ├── model.py
    └── visualize.py

```

# Use Case
Companies may want to provide recommendations for products outside a user’s typical recommendations in order to increase sales by getting current customers to buy additional items not previously purchased.

# Models
ALS recommendation engine
HCA

Libraries
Numpy 
Pandas
Pyspark
Scipy
Sklearn
Pyspark ML

# Minimal Viable Product
Recommender that can return 20 recommendations, 10 from ALS and 10 from echo chamber model.

# Three Difficulties
1.	How to run the models on AWS
2.	How to productionize the models as a web application
3.	How to get linter to work in Atom

# Action plan/schedule 
## Week 1
Day 1 - Project Proposals

Day 2 - First Stinky Model: based on the reduced version of the dataset. The model will include the ALS recommender and the HCA

Day 3 -  Expand FSM: Get FSM to provide recommendations from nearest clusters.

## Week 2

Day 1 – AWS Implementation: Adapt code to full dataset and get code running on AWS

Day 2 - Minimum viable Product 

Day 3 – User Input: Adapt/develop user input code

Day 4 - Substantial Completion, First Practice Presentation: Develop web application

Day 5 – Finish Product: Finish developing web application

## Week 3 

Day 1 	- Documentation Review & Feedback

Day 2   - Document Review & Practice Presentations

Day 3 	- Documentation Review & Dress rehearsal/present to class

Day 4 - Data Science Project Expo

# References
F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History and Context. ACM Transactions on Interactive Intelligent Systems (TiiS) 5, 4, Article 19 (December 2015), 19 pages. DOI=http://dx.doi.org/10.1145/2827872

