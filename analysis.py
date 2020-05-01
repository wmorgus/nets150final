import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def avg_sentiment_by_source(csv, term):
    """
    For each news source, extracts relevant tweets and calculates the average sentiment of relevant tweets

    :param csv: name of the csv
    :param term: relevant term in tweets
    :return: bias value, df of avg sentiment of relevant tweets by news source
    """
    df = pd.read_csv(csv)

    df.dropna(inplace=True)  # drop rows with null text
    df.drop(['status_id', 'created_at'], axis=1, inplace=True)  # remove irrelevant columns

    term_df = df[df['text'].str.contains(term)]
    term_avg_df = term_df.groupby('screen_name').mean()

    return df['bias'][0], list(term_avg_df['sent'])


if __name__ == '__main__':
    terms = ['trump', 'republican', 'democrat', 'american']
    csvs = ['very_left.csv', 'leans_left.csv', 'central.csv', 'leans_right.csv', 'very_right.csv']
    bias_labels = ['Very Left', 'Leans Left', 'Central', 'Leans Right', 'Very Right']

    for term in terms:
        sents = [avg_sentiment_by_source(csv, term)[1] for csv in csvs]

        # box plot
        fig1, ax1 = plt.subplots()
        ax1.set_title(f'"{term.capitalize()}" Tweets: Average Sentiment by News Source')
        ax1.boxplot(sents)

        plt.xticks(list(range(1, len(bias_labels) + 1)), bias_labels)
        plt.xlabel('Political Bias')
        plt.ylabel('Sentiment')

        plt.show()

        # line graph
        x_pos = np.arange(len(sents))
        plt.plot(x_pos, [np.mean(l) for l in sents], marker='o', linewidth=2, markersize=6)

        plt.xticks(x_pos, bias_labels)
        plt.xlabel('Political Bias')
        plt.ylabel('Sentiment')
        plt.title(f'"{term.capitalize()}" Tweets: Average Sentiment by Political Bias')

        plt.show()
