import streamlit as st
from utils.main import text_preprocessing, predict_sentiment

# Streamlit app for Twitter Sentiment Analysis
def main():
    st.title("Twitter Sentiment Analysis")
    st.write("This app allows you to analyze the sentiment of tweets using machine learning models.")
    
    user_input = st.text_area("Tweet:")

    if st.button("Predict Sentiment"):
        if user_input:
            cleaned_text = text_preprocessing(user_input)
            sentiment = predict_sentiment(cleaned_text) 

            st.write(f"Predicted Sentiment: {sentiment}")
        else:
            st.write("Please enter a tweet to analyze.")

if __name__ == "__main__":
    main()
