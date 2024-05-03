import pandas as pd


dataset_filepath = "datasets/tripadvisor_hotel_reviews.csv"


def read_review_data():
    df = pd.read_csv(dataset_filepath)
    rating_counts = df['Rating'].value_counts()
    print(rating_counts)


read_review_data()
