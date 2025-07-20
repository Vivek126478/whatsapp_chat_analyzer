import streamlit as st
import preprocessor
import helper
import matplotlib.pyplot as plt
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






