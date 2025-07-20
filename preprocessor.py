import streamlit as st
import pandas as pd
import re

def preprocess_data(file_data):
    # Normalize special spaces from WhatsApp export (narrow no-break space U+202F)
    file_data = file_data.replace('\u202f', ' ')

    # Regex to capture date/time and message (system or user)
    pattern = r'(\d{1,2}/\d{1,2}/\d{2,4}, \d{1,2}:\d{2} [APMapm]{2}) - (.*?)(?=\n\d{1,2}/\d{1,2}/\d{2,4}, \d{1,2}:\d{2} [APMapm]{2} - |\Z)'
    messages = re.findall(pattern, file_data, re.DOTALL)

    dates = []
    texts = []

    for date, text in messages:
        dates.append(date)
        texts.append(text.strip())

    df = pd.DataFrame({'raw_message': texts, 'datetime': dates})

    # Parse datetime with error coercion
    df['datetime'] = pd.to_datetime(df['datetime'], format='%m/%d/%y, %I:%M %p', errors='coerce')

    users = []
    messages = []

    # Split user and message or assign "Group Notification"
    for msg in df['raw_message']:
        parts = re.split(r'^([^:]+): ', msg, maxsplit=1)
        if len(parts) == 3:
            users.append(parts[1])
            messages.append(parts[2])
        else:
            users.append("Group Notification")
            messages.append(parts[0])

    df['user'] = users
    df['message'] = messages
    df.drop(columns=['raw_message'], inplace=True)
    df['year'] = df['datetime'].dt.year
    df['month_num'] = df['datetime'].dt.month
    df['month'] = df['datetime'].dt.month_name()
    df['only_date'] = df['datetime'].dt.date
    df["day_name"] = df['datetime'].dt.day_name()

    return df


