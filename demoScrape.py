import requests
import tldextract
from bs4 import BeautifulSoup

url = "https://www.cooperative.com/Pages/default.aspx"

html = requests.get(url)

s = BeautifulSoup(html.content, 'html.parser')

results = s.find(id='ResultsContainer')

links = results.find_all('h2', class_='title is-5')

