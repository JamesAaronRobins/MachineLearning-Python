
import warnings

import requests
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

NEWS_SOURCES = ['https://feeds.bbci.co.uk/news/rss.xml', 
               'https://feeds.skynews.com/feeds/rss/home.xml', 
               'https://www.theguardian.com/uk/rss', 
               'https://www.gbnews.com/feeds/feed.rss']

SOURCE_LIST  = ['BBC', 'SkyNews', 'TheGuardian', 'GBNews']


def get_soup(url):
	newspage = requests.get(url, timeout=10)
	soup = BeautifulSoup(newspage.text, "html.parser")	
	return soup


def get_headlines(news_webpage, total_headlines, news_site):
    soup = get_soup(news_webpage)
    
    headlines = soup.find_all('item')
    
    for idx, item in enumerate(headlines[:total_headlines], start=1):
        try:
            title = item.title.text
            print(idx, title)
        except AttributeError:
            # Missing <title> or unexpected structure
            print(f"Only {idx - 1} headlines from {news_site}")
            break

def main():
    for url, source in zip(NEWS_SOURCES, SOURCE_LIST):
        print(f"Starting news headlines from {source}")
        get_headlines(url, 15, source)
        print(f"Finished news from {source}\n---------------------")

if __name__ == "__main__":
    main()