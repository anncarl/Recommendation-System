from flask import Flask, render_template, url_for, request, Response

import numpy as np
import pandas as pd

app = Flask(__name__)

# load the models
location_df = pd.read_pickle("./objects/dataframe.pkl")
kmeans_model = pd.read_pickle("./objects/kmeans_model.pkl")

# load models for colab FILTERING
cf_items_matrix = pd.read_pickle("./objects/item_item_matrix.pkl")
cf_preds = pd.read_pickle("./objects/cf_preds.pkl")

# function to get location results
# Creating Location-Based Recommendation Function
def location_based_recommendation(kmeans, df, latitude, longitude):

    """Predict the cluster for longitude and latitude provided"""
    cluster = kmeans.predict(np.array([latitude,longitude]).reshape(1,-1))[0]
    print("This restaurant belongs to cluster:", cluster)

    """Get the best restaurant in this cluster along with the relevant information for a user to make a decision"""
    print(df.columns)
    return df[df['cluster']==cluster].iloc[0:10][['name', 'categories','stars']]


#creating the collaborative filtering function for restaurant-restaurant recommendation

def cf_recommender(cf_preds_df, item_item_matrix, restaurant):

    """Getting the correlation of a specific restaurant with other Toronto Restaurants"""
    try:
        restaurant_ratings = cf_preds_df.T[restaurant]
        similar_restaurant_ratings = cf_preds_df.T.corrwith(restaurant_ratings)
        corr_ratings = pd.DataFrame(similar_restaurant_ratings, columns=['Correlation'])
        corr_ratings.dropna(inplace=True)

        """Retrieving the Ratings Scores from the Item-Item Matrix"""
        ratings_sim = item_item_matrix[restaurant]

        """Filtering for positively correlated restaurants"""
        ratings_sim = ratings_sim[ratings_sim>0]

        """Generate Top 10 Recommended Restaurants"""
        """Exclude top row as that will be the same restaurant"""
        ff = list(ratings_sim.sort_values(ascending= False).head(11)[1:].index)
        return location_df[location_df['name'].isin(ff)][['name', 'categories','stars']]
        # print(ff)
        # return list(ratings_sim.sort_values(ascending= False).head(11)[1:].index)

    except Exception as e:
        return []




@app.route("/Landing", methods =["GET"])
def index():
    return render_template("Landing.html")



@app.route("/about", methods=["GET"])
def about():
    return render_template("about.html")


@app.route("/location", methods=["GET", "POST"])
def location():
    if request.method =="POST":
        data_send = request.form.to_dict(flat=True)
        results = location_based_recommendation(kmeans_model, location_df, data_send['latitude'], data_send['longitude'])
        results = [list(values) for values in results.values]
        return render_template("location.html", results= results, is_results=True)

    return render_template("location.html")


@app.route("/collaborative2", methods=["GET", "POST"])
def collaborative():

    if request.method =="POST":
        data_send = request.form.to_dict(flat=True)
        try:
            results = cf_recommender(cf_preds, cf_items_matrix, data_send['restaurant'])
        except Exception as e:
            pass

        if len(results)>0 :
            status =True
            results = [list(values) for values in results.values]
        else:
            results =[]
            status =False
        return render_template("collaborative2.html", status=status, results=results, is_results=True)
    return render_template("collaborative2.html")


if __name__ == "__main__":
    app.run(debug=True)
