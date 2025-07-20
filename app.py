import streamlit as st
import preprocessor
import helper
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from typing import cast,Any
import helper
st.sidebar.title("WhatsApp Chat Analyzer")

upload_file = st.sidebar.file_uploader("Upload WhatsApp Chat File", type=["txt"])
if upload_file is not None:
    bytes_data = upload_file.getvalue()
    data = bytes_data.decode("utf-8")
    st.sidebar.success("File uploaded successfully!")
    df = preprocessor.preprocess_data(data)
    st.success("✅ File processed successfully!")      # Basic table view      
    
    #fetch unique users
    user_list = df["user"].unique().tolist()
    user_list.remove("Group Notification")  # Remove system messages if present
    user_list.sort()
    user_list.insert(0, "Overall")

    selected_user = st.sidebar.selectbox("Show Analysis for user", user_list)

    if st.sidebar.button("Show Analysis:"):
        num_messages, words, num_media_messages, num_links = helper.fetch_stats(selected_user,df)
        st.title("Chat Analysis")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.subheader("Total Messages")
            st.title(num_messages)
        with col2:
            st.header("Total Words")
            st.title(words)
        with col3:
            st.subheader("Total Media Messages")
            st.title(num_media_messages)
        with col4:
            st.subheader("Total Links")
            st.title(num_links)
        
        # The busiest users in the group
        if selected_user == "Overall":
            st.title("Most Interactive Users")
            x,new_df = helper.most_interactive_users(df)
            fig, ax = plt.subplots()
            col1 ,col2 = st.columns(2)
            with col1:
                ax.bar(x.index.astype(str), x.to_numpy(),color="red")
                plt.xticks(rotation='vertical')
                st.pyplot(fig)
            with col2:
                st.dataframe(new_df)

        #Wordcloud
        df_wc = helper.create_wordcloud(selected_user, df)
        st.title("Word Cloud")
        fig, ax = plt.subplots()
        ax.imshow(df_wc)
        st.pyplot(fig)

        #most common words
        most_common_df = helper.most_common_words(selected_user,df)
        st.title("Most Common Words")
        fig,axis = plt.subplots()
        axis.barh(most_common_df["Word"], most_common_df["Count"], color="green")
        plt.xticks(rotation='vertical')
        st.pyplot(fig)
        st.write("Top 20 most common words used by the selected user:")
        st.dataframe(most_common_df)

        # emoji analysis
        emoji_df = helper.emoji_helper(selected_user, df)

        if not emoji_df.empty:
            col1, col2 = st.columns(2)

            with col1:
                st.title("Emoji Analysis")
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.bar(emoji_df["Emoji"].astype(str), emoji_df["Count"], color="skyblue")
                ax.set_xlabel("Emoji", fontsize=14)
                ax.set_ylabel("Count", fontsize=14)
                ax.set_title("Top Emojis", fontsize=16)
                plt.xticks(fontsize=18)
                st.pyplot(fig)

            with col2:
                st.dataframe(emoji_df)
        else:
            st.warning("No emojis found for the selected user.")

        # Monthly timeline
        st.title("Monthly Timeline")
        timeline = helper.monthly_timeline(selected_user, df)
        fig, ax = plt.subplots(figsize=(12, 6))
        plt.plot(timeline['time'], timeline['message_count'], marker='o', color='purple')
        plt.xticks(rotation="vertical")
        plt.xlabel("Month-Year")
        plt.ylabel("Messages")
        plt.title("Monthly Message Timeline")
        st.pyplot(fig)

        # Daily timeline
        st.title("Daily Timeline")
        daily_timeline = helper.daily_timeline(selected_user, df)
        fig, ax = plt.subplots(figsize=(12, 6))
        plt.plot(daily_timeline['only_date'], daily_timeline['message'], marker='o', color='orange')
        plt.xticks(rotation="vertical")
        plt.xlabel("Date")
        plt.ylabel("Messages")
        plt.title("Daily Message Timeline")
        st.pyplot(fig)

        # Activity map
        st.title("Weekly Activity Map")
        col1 ,col2 = st.columns(2)
        with col1:
            st.subheader("Most Active Day")
            activity_map = helper.weekly_activity_map(selected_user, df)
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.bar(activity_map["day_name"], activity_map["message_count"], color="blue")
            plt.xticks(rotation='vertical')
            st.pyplot(fig)
        with col2:
            st.subheader("Most Active Month")
            month_activity = helper.month_activity_map(selected_user, df)
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.bar(month_activity["month"], month_activity["message_count"], color="green")
            plt.xticks(rotation='vertical')
            st.pyplot(fig)
        # Build interaction graph and get centrality metrics
        G, degree_centrality, betweenness_centrality, closeness_centrality = helper.build_interaction_graph(df)

        st.title("User Interaction Network")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Interaction Graph")
            fig, ax = plt.subplots(figsize=(10, 7))
            pos = nx.spring_layout(G, k=0.5, seed=42)
            
            node_sizes = [5000 * degree_centrality.get(node, 0.01) for node in G.nodes()]
            nx.draw_networkx_nodes(G, pos, node_size=cast(Any, node_sizes), node_color='skyblue', alpha=0.9, ax=ax)
            
            edges = G.edges(data=True)
            edge_weights = [edata.get('weight', 1) for _, _, edata in edges]
            edge_widths = [max(w / 2, 0.5) for w in edge_weights]
            nx.draw_networkx_edges(G, pos, width=cast(Any, edge_widths), alpha=0.7, edge_color='grey', arrowsize=20, ax=ax)
            
            nx.draw_networkx_labels(G, pos, font_size=10, font_color='black', ax=ax)
            
            ax.set_title("User Interaction Graph (Edges = consecutive replies)", fontsize=14)
            ax.axis('off')
            st.pyplot(fig)

        with col2:
            st.subheader("User Centrality Scores")
            centrality_df = pd.DataFrame({
                "User": list(G.nodes()),
                "Degree Centrality": [degree_centrality.get(u, 0) for u in G.nodes()],
                "Betweenness Centrality": [betweenness_centrality.get(u, 0) for u in G.nodes()],
                "Closeness Centrality": [closeness_centrality.get(u, 0) for u in G.nodes()]
            }).sort_values(by="Degree Centrality", ascending=False)
            st.dataframe(centrality_df)
            
        # Topic Modeling Section
    st.title("Chat Topic Discovery (LDA)")
    
    # Unpack topics, lda_model, dictionary, corpus
    topics, lda_model, dictionary, corpus = helper.extract_topics(selected_user, df)

    if topics:
        for topic in topics:
            st.write(topic)
    else:
        st.warning("Not enough data for topic modeling.")

    st.success("Analysis completed successfully!")

    # Topic distribution plot
    if selected_user and topics:
        st.subheader(f"Topic Distribution for {selected_user}")
        topic_dist_df = helper.get_user_topic_distribution(selected_user, df, lda_model, dictionary, corpus)

        fig, ax = plt.subplots(figsize=(10,5))
        ax.bar(topic_dist_df["Topic"], topic_dist_df["Weight"], color="purple")
        ax.set_ylabel("Average Topic Weight")
        ax.set_xlabel("Topics")
        ax.set_title(f"Topic Distribution for {selected_user}")
        st.pyplot(fig)

        # Sentiment Analysis Section
        st.title("Sentiment Analysis")

        sentiment_counts, monthly_sentiment = helper.sentiment_analysis(selected_user, df)

        # Pie chart for overall sentiment distribution
        fig1, ax1 = plt.subplots()
        ax1.pie(sentiment_counts['Count'], labels=sentiment_counts['Sentiment'].tolist(), autopct='%1.1f%%', 
        startangle=90, colors=['green', 'red', 'grey'])
        ax1.axis('equal')  # Equal aspect ratio ensures pie is a circle.
        st.subheader(f"Overall Sentiment Distribution for {selected_user}")
        st.pyplot(fig1)

        # Line chart for monthly sentiment trend
        fig2, ax2 = plt.subplots(figsize=(10,5))
        ax2.plot(monthly_sentiment['time'], monthly_sentiment['avg_compound'], marker='o', linestyle='-', color='blue')
        plt.xticks(rotation=45)
        plt.xlabel("Month-Year")
        plt.ylabel("Average Compound Sentiment Score")
        plt.title(f"Monthly Sentiment Trend for {selected_user}")
        st.pyplot(fig2)

        st.title("Topic Trends Over Time")

        topic_time_df = helper.get_topic_distribution_over_time(selected_user, df, lda_model, dictionary)

        fig, ax = plt.subplots(figsize=(12, 6))

        for topic in topic_time_df.columns:
            if topic.startswith("Topic"):
                ax.plot(topic_time_df["time"], topic_time_df[topic], marker='o', label=topic)

        ax.set_xlabel("Time")
        ax.set_ylabel("Average Topic Weight")
        ax.set_title(f"Topic Trends Over Time for {selected_user}")
        ax.legend()
        plt.xticks(rotation=45)
        st.pyplot(fig)

        #keywords over time
        st.title("Keyword Trend Analysis Over Time")

        # Get top common words for selected user
        top_common_words_df = helper.most_common_words(selected_user, df)
        top_keywords = top_common_words_df["Word"].tolist()[:5]  # take top 5 keywords for trend

        # Get trend data from helper function
        trend_df = helper.keyword_trend_analysis(selected_user, df, top_keywords)

        if trend_df.empty:
            st.write("Not enough data to show keyword trends.")
        else:
            fig, ax = plt.subplots(figsize=(12, 6))
            for kw in top_keywords:
                ax.plot(trend_df['time'], trend_df[kw], marker='o', label=kw)

            ax.set_title(f"Keyword Trends for {selected_user}")
            ax.set_xlabel("Time")
            ax.set_ylabel("Message Count")
            ax.legend(title="Keywords")
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)
            
        # Conversation summary
        summary = helper.conversation_summarization(selected_user, df)
        st.title("Conversation Summary")
        for i, sentence in enumerate(summary, 1):
            st.write(f"{i}. {sentence}")

        # Named Entity Recognition
        ner_df = helper.named_entity_recognition(selected_user, df)
        st.title("Named Entities Extracted")
        st.dataframe(ner_df.head(20))

        # Emotion Detection
        emotion_df = helper.emotion_detection(selected_user, df)
        st.title("Emotion Detection")
        st.bar_chart(emotion_df.set_index("Emotion")["Count"])











