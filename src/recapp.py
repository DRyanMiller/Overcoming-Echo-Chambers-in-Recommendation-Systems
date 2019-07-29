from src import custom as cm


def recommendations():
    user_ratings = cm.user_rankings()
    user_fac = cm.user_factors(user_ratings)
    ALS_recs = cm.ALS_recommendations(user_fac)
    user_cluster = cm.get_user_cluster(user_fac)
    new_recs = cm.get_recommendations(user_cluster)
    print('\n\n\n')
    print('Try:')
    for movie in ALS_recs: print('    {}'.format(movie), sep = "\n")
    print('\n')
    print('You may also like:')
    for movie in new_recs: print('    {}'.format(movie), sep = "\n")
