
<h1> Observations </h1>
<ol>
    <li> Sentiment of each tweet from the Media outlets is very random.  Not much dependency on the prior tweet. </li>
    <li> In general there were lot of neutral sentiment tweet from each Media outlet. </li>
    <li> New York Times and CNN had on an average more negetive sentiment compared to other Media outlets.</li>

</ol> 


```python
# Dependencies
import tweepy
import numpy as np
import pandas as pd
from pprint import pprint
import matplotlib.pyplot as plt
import seaborn as sns

# Import and Initialize Sentiment Analyzer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()

# Twitter API Keys
from config import (consumer_key, 
                    consumer_secret, 
                    access_token, 
                    access_token_secret)

# Setup Tweepy API Authentication
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth, parser=tweepy.parsers.JSONParser())
```


```python
# Target Search Term
target_terms = ("@BBC", "@CBS", "@CNN",
                "@Fox", "@NewYorkTimes")



```


```python
# List to hold results
results_list = []

# "Real Person" Filters
min_tweets = 5
max_tweets = 10000
max_followers = 2500
max_following = 2500
lang = "en"

# Loop through all target users
for target in target_terms:
    #variable to hold tweets ago
    tweet_ago = 0
    oldest_tweet = None
    if (tweet_ago < 100):
        # Loop through 20 times
        for x in range(20):

            # Run search around each tweet
            public_tweets = api.search(
                target, count=100, result_type="recent", max_id=oldest_tweet)

            # Loop through all tweets
            for tweet in public_tweets["statuses"]:
                # Use filters to check if user meets conditions
                if (tweet["user"]["followers_count"] < max_followers
                    and tweet["user"]["statuses_count"] > min_tweets
                    and tweet["user"]["statuses_count"] < max_tweets
                    and tweet["user"]["friends_count"] < max_following
                    and tweet["user"]["lang"] == lang
                    and tweet_ago < 100):

                    # Run Vader Analysis on each tweet
                    results = analyzer.polarity_scores(tweet["text"])
                    compound = results["compound"]
                    pos = results["pos"]
                    neu = results["neu"]
                    neg = results["neg"]
                    
                    sentiment = {
                        "User": target,
                        "Compound": compound,
                        "Positive": pos,
                        "Neutral": neu,
                        "Negative": neg,
                        "Tweet_Ago": tweet_ago,
                        "Tweet_Text": tweet["text"],
                        "Tweet_DateTime" : tweet['created_at']
                    }
                    results_list.append(sentiment)
                    tweet_ago += 1
                    
            oldest_tweet = tweet["id"] - 1
        
```


```python
df = pd.DataFrame(results_list)
df.to_csv("Output/MediaSentiment.csv", index=False, header=True)
```


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Compound</th>
      <th>Negative</th>
      <th>Neutral</th>
      <th>Positive</th>
      <th>Tweet_Ago</th>
      <th>Tweet_DateTime</th>
      <th>Tweet_Text</th>
      <th>User</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>0</td>
      <td>Sun Jun 10 00:37:18 +0000 2018</td>
      <td>@GStarFreedom @CarlWil42543044 @Karenco30 @lor...</td>
      <td>@BBC</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.0000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>1</td>
      <td>Sun Jun 10 00:36:46 +0000 2018</td>
      <td>RT @KISSfanaticz: @GGJuliePayette @RoyalFamily...</td>
      <td>@BBC</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.6597</td>
      <td>0.316</td>
      <td>0.684</td>
      <td>0.000</td>
      <td>2</td>
      <td>Sun Jun 10 00:36:33 +0000 2018</td>
      <td>RT @Trickyjabs: Weird isn't it:\n\nPoll Tax Pr...</td>
      <td>@BBC</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.4939</td>
      <td>0.000</td>
      <td>0.814</td>
      <td>0.186</td>
      <td>3</td>
      <td>Sun Jun 10 00:36:32 +0000 2018</td>
      <td>RT @Majid_Agha: H A R D  T A L K @BBC with @Th...</td>
      <td>@BBC</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.3612</td>
      <td>0.147</td>
      <td>0.791</td>
      <td>0.062</td>
      <td>4</td>
      <td>Sun Jun 10 00:36:27 +0000 2018</td>
      <td>RT @BBC: These turtles have to battle a wall o...</td>
      <td>@BBC</td>
    </tr>
  </tbody>
</table>
</div>




```python

sns.lmplot(x="Tweet_Ago", y="Compound", data=df, hue="User", fit_reg=False,  
           scatter_kws={ 'edgecolor':'black'}, legend=False);

plt.xlim(105,-5)
plt.ylim(-1.05,1.05)
plt.xlabel("Tweets Ago")
plt.ylabel("Tweet Polarity")
plt.legend(title="Media Sources", bbox_to_anchor=(1.05,1),loc=2)
plt.title("Sentiment Analysis of Media Tweets")
plt.grid()
plt.savefig("Output/SentimentAnalysisOfMediaTweets.png")
```


![png](output_6_0.png)



```python
user_df = df.groupby('User')['Compound'].mean().reset_index()
```


```python
user_df.columns = ["User","Polarity_mean"]
user_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>User</th>
      <th>Polarity_mean</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>@BBC</td>
      <td>0.002077</td>
    </tr>
    <tr>
      <th>1</th>
      <td>@CBS</td>
      <td>0.171889</td>
    </tr>
    <tr>
      <th>2</th>
      <td>@CNN</td>
      <td>-0.096477</td>
    </tr>
    <tr>
      <th>3</th>
      <td>@Fox</td>
      <td>0.131302</td>
    </tr>
    <tr>
      <th>4</th>
      <td>@NewYorkTimes</td>
      <td>-0.068120</td>
    </tr>
  </tbody>
</table>
</div>




```python
my_colors = 'rgkymcb'

plt.title('Overall Media Sentiment based on Twitter')
plt.ylabel('Mean Tweet Compound Polarity')
plt.xlabel('Media Source')
plt.grid()
plt.bar(user_df['User'],user_df['Polarity_mean'],color=my_colors);
plt.savefig("Output/OverAllSentimentMedia.png")
```


![png](output_9_0.png)

