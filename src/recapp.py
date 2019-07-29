from src import custom as cm


def get_recommendations():
    """Takes in user ratings and returns a list of
    ten ALS recommendations and ten recommendations from
    the two clusters nearest to the user's cluster."""
    user_ratings = cm.get_user_ratings()
    user_fac = cm.get_user_factors(user_ratings)
    ALS_recs = cm.get_als_recommendations(user_fac)
    user_cluster = cm.get_user_cluster(user_fac)
    new_recs = cm.get_new_recommendations(user_cluster)
    print('\n\n\n')
    print('Try:')
    for movie in ALS_recs: print('    {}'.format(movie), sep = "\n")
    print('\n')
    print('You may also like:')
    for movie in new_recs: print('    {}'.format(movie), sep = "\n")
