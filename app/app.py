from fastapi import FastAPI
from requests import request
from pydantic import BaseModel
import uvicorn


from utils.main import text_preprocessing, predict_sentiment

app = FastAPI()

class Tweet(BaseModel):
    text: str

class TextResponse(BaseModel):
    text: str
    predicted_sentiment: str

@app.get("/")
def read_root():
    return {"message": "Welcome to the Twitter Sentiment Analysis API!"}

@app.post("/analyze")
def analyze_sentiment(tweet: Tweet):
    text = tweet.text
    cleaned_text = text_preprocessing(text)
    sentiment = predict_sentiment(cleaned_text)
    return TextResponse(text=text, predicted_sentiment=sentiment)

if __name__ == "__main__":
    uvicorn.run(app, port=8900, reload=True)
