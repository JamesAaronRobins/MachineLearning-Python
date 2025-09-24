
import requests 
from bs4 import BeautifulSoup
from bs4 import XMLParsedAsHTMLWarning
import warnings

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

newssources = ['https://feeds.bbci.co.uk/news/rss.xml', 'https://feeds.skynews.com/feeds/rss/home.xml',
               'https://www.theguardian.com/uk/rss','https://www.gbnews.com/feeds/feed.rss']

SourceList = ['BBC', 'SkyNews', 'TheGuardian', 'GBNews']

def get_soup(url):
	newspage = requests.get(url)
	soup = BeautifulSoup(newspage.text, "html.parser")	
	return soup

def get_headlines(newswebpage, totalheadlines,newssite):
    soup = get_soup(newswebpage)
    
    headlines = soup.find_all('item')
    
    for newsstories in range(totalheadlines):
        try:
            articleheadline = headlines[newsstories].title.text
            print(newsstories, articleheadline)
        except:
            print(f'Only {newsstories} headlines from {newssite}')
            break
        
for i,j in zip(newssources,SourceList):
    print(f'Starting news headlines from {j}')
    a = get_headlines(i, 15,j)
    print(f'Finished news from {j}\n ---------------------')