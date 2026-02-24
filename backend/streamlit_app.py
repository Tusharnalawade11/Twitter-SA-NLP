import streamlit as st
from utils.main import text_preprocessing, predict_sentiment

# Streamlit app for Twitter Sentiment Analysis
def main():

    st.title("Vichaar Sentiment Analysis")
    st.write("This app allows you to analyze the sentiment of vichaar using machine learning models.")

    # Initialize session state for feedback flow
    if "show_feedback" not in st.session_state:
        st.session_state.show_feedback = False
    if "predicted_sentiment" not in st.session_state:
        st.session_state.predicted_sentiment = None
    if "current_tweet" not in st.session_state:
        st.session_state.current_tweet = None

    user_input = st.text_area("Tweet:")

    if st.button("Predict Sentiment"):
        if user_input:
            cleaned_text = text_preprocessing(user_input)
            sentiment = predict_sentiment(cleaned_text)

            st.session_state.predicted_sentiment = sentiment
            st.session_state.current_tweet = user_input
            st.session_state.show_feedback = False
        else:
            st.write("Please enter a tweet to analyze.")

    if st.session_state.predicted_sentiment:
        st.write(f"Predicted Sentiment: {st.session_state.predicted_sentiment}")
        st.write("Is this prediction correct?")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Yes"):
                st.write("Thank you for your confirmation!")
                st.session_state.predicted_sentiment = None
                st.rerun()
        with col2:
            if st.button("No"):
                st.session_state.show_feedback = True

    if st.session_state.show_feedback:
        st.write("Please provide feedback to help improve the model.")
        correct_sentiment = st.text_input("Correct Sentiment")
        if correct_sentiment:
            misclassified_tweets.append({
                "tweet": st.session_state.current_tweet,
                "predicted_sentiment": st.session_state.predicted_sentiment,
                "actual_sentiment": correct_sentiment
            })
            st.write("Thank you for your feedback!")
            st.session_state.predicted_sentiment = None
            st.session_state.show_feedback = False
            print("Misclassified Tweets:", misclassified_tweets)
            st.rerun()


if __name__ == "__main__":
    misclassified_tweets = []
    main()
