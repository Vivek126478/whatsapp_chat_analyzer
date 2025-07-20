from urlextract import URLExtract
from wordcloud import WordCloud
import pandas as pd
from collections import Counter
import emoji
import networkx as nx
import matplotlib.pyplot as plt
import gensim
from gensim import corpora
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('vader_lexicon')
import re
import numpy as np
import spacy
from nrclex import NRCLex
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
def fetch_stats(selected_user, df):
    if selected_user != "Overall":
        df = df[df["user"] == selected_user]
    num_messages = df.shape[0]
    words = []
    for message in df["message"]:
        words.extend(message.split())

    num_media_messages = df[df["message"] == "<Media omitted>"].shape[0]

    extractor = URLExtract()
    links = []
    for message in df["message"]:
        links.extend(extractor.find_urls(message))
    num_links = len(links)

    return num_messages, len(words), num_media_messages, num_links

def most_interactive_users(df):
    x = df['user'].value_counts().head()
    df_percent = round((df["user"].value_counts() / df.shape[0]) * 100, 2).reset_index().rename(
        columns={"index": "user", "user": "Percentage"}
    )
    return x, df_percent  

def create_wordcloud(selected_user, df):
    with open("stop_hinglish.txt", "r") as f:
        stop_words = set(f.read().split())
    if selected_user != "Overall":
        df = df[df["user"] == selected_user]
    temp = df[df["user"] != "Group Notification"]
    temp = temp[temp["message"] != "<Media omitted>"]

    def remove_stop_words(message):
        return " ".join([word for word in message.lower().split() if word not in stop_words])

    wc = WordCloud(width=500, height=500, min_font_size=10, background_color="white")
    temp["message"] = temp["message"].apply(remove_stop_words)
    df_wc = wc.generate(" ".join(temp["message"]))
    return df_wc

def most_common_words(selected_user, df):
    with open("stop_hinglish.txt", "r") as f:
        stop_words = set(f.read().split())

    if selected_user != "Overall":
        df = df[df["user"] == selected_user]

    temp = df[df["user"] != "Group Notification"]
    temp = temp[temp["message"] != "<Media omitted>"]

    words = []
    for message in temp["message"]:
        for word in message.lower().split():
            word = word.strip()
            if word not in stop_words and word != "":
                words.append(word)

    return_df = pd.DataFrame(Counter(words).most_common(20))
    return_df.rename(columns={0: "Word", 1: "Count"}, inplace=True)

    return return_df

def emoji_helper(selected_user, df):
    if selected_user != "Overall":
        df = df[df["user"] == selected_user]

    emojis = []
    for message in df["message"]:
        emojis.extend([c for c in message if c in emoji.EMOJI_DATA])

    emoji_df = pd.DataFrame(Counter(emojis).most_common(5))
    emoji_df.rename(columns={0: "Emoji", 1: "Count"}, inplace=True)

    return emoji_df

def monthly_timeline(selected_user, df):
    if selected_user != "Overall":
        df = df[df["user"] == selected_user]

    timeline = df.groupby(['year', 'month_num', 'month']).count()['message'].reset_index()
    timeline.rename(columns={'message': 'message_count'}, inplace=True)
    timeline['time'] = timeline['month'] + "-" + timeline['year'].astype(str)

    return timeline

def daily_timeline(selected_user, df):
    if selected_user != "Overall":
        df = df[df["user"] == selected_user]

    daily_timeline = df.groupby("only_date").count()["message"].reset_index()
    return daily_timeline

def weekly_activity_map(selected_user, df):
    if selected_user != "Overall":
        df = df[df["user"] == selected_user]

    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    day_counts = df['day_name'].value_counts().reindex(day_order, fill_value=0).reset_index()
    day_counts.columns = ['day_name', 'message_count']

    return day_counts

def month_activity_map(selected_user, df):
    if selected_user != "Overall":
        df = df[df["user"] == selected_user]

    month_order = ['January', 'February', 'March', 'April', 'May', 'June', 
                   'July', 'August', 'September', 'October', 'November', 'December']
    month_counts = df['month'].value_counts().reindex(month_order, fill_value=0).reset_index()
    month_counts.columns = ['month', 'message_count']

    return month_counts

def build_interaction_graph(df):
    df = df[df["user"] != "Group Notification"].reset_index(drop=True)

    interactions = Counter()
    for i in range(len(df) - 1):
        user_from = df.loc[i, "user"]
        user_to = df.loc[i + 1, "user"]
        if user_from != user_to:
            interactions[(user_from, user_to)] += 1

    G = nx.DiGraph()
    for (user_from, user_to), weight in interactions.items():
        G.add_edge(user_from, user_to, weight=weight)

    degree_centrality = nx.degree_centrality(G)
    betweenness_centrality = nx.betweenness_centrality(G)
    closeness_centrality = nx.closeness_centrality(G)

    return G, degree_centrality, betweenness_centrality, closeness_centrality

def extract_topics(selected_user, df, num_topics=5):
    if selected_user != "Overall":
        df = df[df["user"] == selected_user]

    temp = df[df["message"] != "<Media omitted>"]
    stop_words = set(stopwords.words("english"))

    processed_docs = []
    for doc in temp["message"]:
        tokens = [word.lower() for word in doc.split() if word.isalpha() and word.lower() not in stop_words]
        if tokens:
            processed_docs.append(tokens)

    if len(processed_docs) == 0:
        return []

    dictionary = corpora.Dictionary(processed_docs)
    corpus = [dictionary.doc2bow(doc) for doc in processed_docs]

    if len(dictionary) < num_topics:
        return []

    lda_model = gensim.models.LdaModel(corpus=corpus,
                                       id2word=dictionary,
                                       num_topics=num_topics,
                                       random_state=42,
                                       passes=10)

    topics = []
    for idx, topic in lda_model.print_topics(-1):
        topics.append(f"Topic {idx+1}: " + topic)

    return topics, lda_model, dictionary, corpus

def get_user_topic_distribution(selected_user, df, lda_model, dictionary, corpus):
    if selected_user != "Overall":
        user_msgs = df[df["user"] == selected_user]["message"].tolist()
    else:
        user_msgs = df["message"].tolist()

    topic_weights = np.zeros(lda_model.num_topics)
    total_messages = len(user_msgs)
    if total_messages == 0:
        return pd.DataFrame(columns=["Topic", "Weight"])

    for message in user_msgs:
        bow = dictionary.doc2bow(message.lower().split())
        topics = lda_model.get_document_topics(bow)
        for topic_num, weight in topics:
            topic_weights[topic_num] += weight

    topic_weights /= total_messages

    df_topic_dist = pd.DataFrame({
        "Topic": [f"Topic {i+1}" for i in range(lda_model.num_topics)],
        "Weight": topic_weights
    })

    return df_topic_dist

def sentiment_analysis(selected_user, df):
    sid = SentimentIntensityAnalyzer()

    if selected_user != "Overall":
        df = df[df["user"] == selected_user]

    df = df[df["message"] != "<Media omitted>"].copy()

    df['sentiment_compound'] = df['message'].apply(lambda msg: sid.polarity_scores(msg)['compound'])

    def categorize_sentiment(score):
        if score >= 0.05:
            return "Positive"
        elif score <= -0.05:
            return "Negative"
        else:
            return "Neutral"

    df['sentiment_category'] = df['sentiment_compound'].apply(categorize_sentiment)

    sentiment_counts = df['sentiment_category'].value_counts().reset_index()
    sentiment_counts.columns = ['Sentiment', 'Count']

    if 'year' not in df.columns or 'month' not in df.columns:
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month_name()

    monthly_sentiment = df.groupby(['year', 'month']).agg(
        avg_compound=('sentiment_compound', 'mean'),
        message_count=('sentiment_compound', 'count')
    ).reset_index()

    monthly_sentiment['time'] = monthly_sentiment['month'] + "-" + monthly_sentiment['year'].astype(str)

    return sentiment_counts, monthly_sentiment[['time', 'avg_compound']]

def get_topic_distribution_over_time(selected_user, df, lda_model, dictionary):

    if selected_user != "Overall":
        df = df[df["user"] == selected_user]

    data = []
    for _, row in df.iterrows():
        message = row["message"]
        tokens = [word.lower() for word in message.split() if word.isalpha()]
        bow = dictionary.doc2bow(tokens)
        topic_probs = lda_model.get_document_topics(bow)

        topic_dist = np.zeros(lda_model.num_topics)
        for topic_num, prob in topic_probs:
            topic_dist[topic_num] = prob

        data.append([row["year"], row["month_num"], row["month"], topic_dist])

    topic_df = pd.DataFrame(data, columns=["year", "month_num", "month", "topic_dist"])

    grouped = topic_df.groupby(["year", "month_num", "month"])["topic_dist"].apply(list).reset_index()

    def sum_vectors(vectors):
        return np.sum(np.vstack(vectors), axis=0)

    grouped["topic_sum"] = grouped["topic_dist"].apply(sum_vectors)

    grouped["message_count"] = grouped["topic_dist"].apply(len)
    grouped["topic_avg"] = grouped.apply(lambda x: x["topic_sum"] / x["message_count"], axis=1)

    topic_cols = [f"Topic {i+1}" for i in range(lda_model.num_topics)]
    topic_weights_df = pd.DataFrame(grouped["topic_avg"].tolist(), columns=topic_cols)

    final_df = pd.concat([grouped[["year", "month_num", "month"]], topic_weights_df], axis=1)

    final_df["time"] = pd.to_datetime(final_df["year"].astype(str) + "-" + final_df["month_num"].astype(str))

    final_df.sort_values(by="time", inplace=True)

    return final_df

def keyword_trend_analysis(selected_user, df, keywords, top_n=10):
    if selected_user != "Overall":
        df = df[df["user"] == selected_user]

    temp = df[(df["user"] != "Group Notification") & (df["message"] != "<Media omitted>")].copy()

    temp['time'] = temp['month'] + "-" + temp['year'].astype(str)
    keyword_counts = {kw: [] for kw in keywords}

    grouped = temp.groupby('time')

    for time_period, group in grouped:
        messages = group['message'].str.lower()
        for kw in keywords:
            count = messages.str.contains(r'\b' + re.escape(kw.lower()) + r'\b').sum()
            keyword_counts[kw].append((time_period, count))

    df_list = []
    for kw, counts in keyword_counts.items():
        df_kw = pd.DataFrame(counts, columns=['time', kw])
        df_list.append(df_kw.set_index('time'))

    trend_df = pd.concat(df_list, axis=1).fillna(0)

    trend_df.index = pd.to_datetime(trend_df.index, format='%B-%Y')
    trend_df = trend_df.sort_index()

    trend_df.reset_index(inplace=True)
    trend_df.rename(columns={'index': 'time'}, inplace=True)

    return trend_df

nlp = spacy.load("en_core_web_sm")

def conversation_summarization(selected_user, df, num_sentences=5):
    """
    Extractive summary of chat messages using TextRank algorithm.
    Returns list of top sentences as summary.
    """

    if selected_user != "Overall":
        df = df[df["user"] == selected_user]

    messages = df["message"].tolist()

    messages = [msg for msg in messages if msg.strip() and msg != "<Media omitted>"]

    if len(messages) <= num_sentences:
        return messages

    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(messages)

    sim_matrix = cosine_similarity(X)

    nx_graph = nx.from_numpy_array(sim_matrix)
    scores = nx.pagerank(nx_graph)

    ranked_sentences = sorted(((scores[i], s) for i, s in enumerate(messages)), reverse=True)

    summary = [s for _, s in ranked_sentences[:num_sentences]]

    return summary

def named_entity_recognition(selected_user, df):
    """
    Extract named entities from chat messages using SpaCy NER.
    Returns a DataFrame with entity and label counts.
    """

    if selected_user != "Overall":
        df = df[df["user"] == selected_user]

    messages = df["message"].tolist()
    messages = [msg for msg in messages if msg.strip() and msg != "<Media omitted>"]

    entities = []

    for message in messages:
        doc = nlp(message)
        for ent in doc.ents:
            entities.append((ent.text, ent.label_))

    df_entities = pd.DataFrame(entities, columns=["Entity", "Label"])
    entity_counts = df_entities.groupby(["Entity", "Label"]).size().reset_index(name="Count")

    entity_counts = entity_counts.sort_values(by="Count", ascending=False)

    return entity_counts

def emotion_detection(selected_user, df):
    """
    Emotion detection using NRC lexicon via NRCLex.
    Returns a DataFrame with emotions and their counts.
    """

    if selected_user != "Overall":
        df = df[df["user"] == selected_user]

    messages = df["message"].tolist()
    messages = [msg for msg in messages if msg.strip() and msg != "<Media omitted>"]

    text_blob = " ".join(messages)

    emotion_obj = NRCLex(text_blob)

    emotions_count = emotion_obj.raw_emotion_scores

    emotions_df = pd.DataFrame(list(emotions_count.items()), columns=["Emotion", "Count"])

    emotions_df = emotions_df.sort_values(by="Count", ascending=False)

    return emotions_df




