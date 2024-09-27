import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.sentiment.vader import SentimentIntensityAnalyzer


results_file_path = r'C:\Users\pc\OneDrive\Desktop\sentiment_analysis\res.xlsx' 
data = pd.read_excel(results_file_path) 


sid = SentimentIntensityAnalyzer()


def sentiment_Vader(text):
    
    if isinstance(text, str) and text:
        scores = sid.polarity_scores(text)
        return scores['compound'] 
    return 0  


data['score'] = data['clean_text'].apply(sentiment_Vader)


def analyze(x):
    if x >= 0.5:
        return 'Positive'
    elif x <= -0.5:
        return 'Negative'
    else:
        return 'Neutral'

data['analysis'] = data['score'].apply(analyze)

data.to_excel(results_file_path, index=False) 


plt.figure(figsize=(10, 6))
sns.countplot(x=data['analysis'], order=data['analysis'].value_counts().index)
plt.title('Sentiment Distribution (Count Plot)')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.savefig('sentiment_distribution_count.png')  
plt.show()

plt.figure(figsize=(8, 8))
sentiment_counts = data['analysis'].value_counts()
plt.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=90, colors=['lightcoral', 'lightskyblue', 'lightgreen'])
plt.title('Sentiment Distribution (Pie Chart)')
plt.axis('equal')  
plt.savefig('sentiment_distribution_pie.png')  
plt.show()

average_scores = data.groupby('analysis')['score'].mean().reset_index()

plt.figure(figsize=(10, 6))
sns.barplot(x='analysis', y='score', data=average_scores, palette='viridis')
plt.title('Average Sentiment Score by Category')
plt.xlabel('Sentiment Category')
plt.ylabel('Average Score')
plt.savefig('average_sentiment_scores.png')  
plt.show()
