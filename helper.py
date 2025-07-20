from urlextract import URLExtract
from wordcloud import WordCloud
import pandas as pd
from collections import Counter
import emoji
def fetch_stats(selected_user, df):

    if selected_user != "Overall":
        # If a specific user is selected, filter the DataFrame
        df = df[df["user"] == selected_user]
    num_messages = df.shape[0]
    # No of words
    words = []
    for message in df["message"]:
        words.extend(message.split())

    #fetch the media
    num_media_messages = df[df["message"] == "<Media omitted>"].shape[0]
    #fetch the links
    links = []
    extractor = URLExtract()
    for message in df["message"]:
        links.extend(extractor.find_urls(message))
    num_links = len(links)


    return num_messages,len(words), num_media_messages, num_links


def most_interactive_users(df):
    x = df['user'].value_counts().head()
    df = round((df["user"].value_counts() / df.shape[0]) * 100, 2).reset_index().rename(
        columns={"index": "user", "user": "Percentage"}
    )
    return x , df  

def create_wordcloud(selected_user, df):
    with open("stop_hinglish.txt", "r") as f:
        stop_words = set(f.read().split())
    if selected_user != "Overall":
        df = df[df["user"] == selected_user]
    temp = df[df["user"] != "Group Notification"]
    temp = temp[temp["message"] != "<Media omitted>"]

    def remove_stop_words(message):
        lst = []
        for word in message.lower().split():
            if word not in stop_words:
                lst.append(word)
        return " ".join(lst)

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

    # Group by year and month_num
    timeline = df.groupby(['year', 'month_num', 'month']).count()['message'].reset_index()
    timeline.rename(columns={'message': 'message_count'}, inplace=True)

    # Create time label like "January-2024"
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

    day_counts = df['day_name'].value_counts().reindex(
        ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
        fill_value=0
    ).reset_index()

    day_counts.columns = ['day_name', 'message_count']

    return day_counts

def month_activity_map(selected_user, df):
    if selected_user != "Overall":
        df = df[df["user"] == selected_user]

    month_counts = df['month'].value_counts().reindex(
        ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'],
        fill_value=0
    ).reset_index()

    month_counts.columns = ['month', 'message_count']

    return month_counts

    


    
