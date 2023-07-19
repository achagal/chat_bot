import requests
import time
import tldextract
from bs4 import BeautifulSoup
from collections import deque
from urllib.parse import urljoin
import os
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import TextLoader
from dotenv import find_dotenv, load_dotenv
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)




os.environ["OPENAI_API_KEY"] = "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"
load_dotenv(find_dotenv())
embeddings = OpenAIEmbeddings()

# load in saved data base that we will be adding documents to
db = FAISS.load_local('U:\Projects\Chat_bot', embeddings, "faiss_index")


# data structures to store need to visit and already visited links
queue = deque()
link_set = set()

# set domain to be bts pages so you dont click on ads or non-bts parts of site

allowed_domain = "cooperative.com"

# this functoin gets all the links from a single page of a website 
# and adds them to the queue if and only if they have not yet been visited
def process_link(page):
    # open url (with error handling)
    try:
        response = requests.get(page)
    except Exception as e:
        print(f"An error occurred while fetching {page}: {e}")
        return

    # create soup object
    soup = BeautifulSoup(response.text, 'html.parser')


    # find all <a> html tags on webpage
    links = soup.find_all('a')

    # add all valid url's as strings to queue 
    for link in links:
        # Check if <i class="fa fa-users" aria-hidden="true"></i> is in the link
        i_tag = link.find('i', attrs={'aria-hidden': 'true'})
        
        # If it is, then continue to next iteration
        if i_tag is not None:
            continue

        # If link does not have an href attribute, skip to next iteration
        if not link.has_attr('href'):
            continue

        # otherwise, extract just url from html container
        url = link.get('href')

        # properly format the url
        url = urljoin(page, url)

        # check domain before joining url and page
        tld = tldextract.extract(url).registered_domain
        # if the site is not on BTS site (an ad or diff sector) skip it
        if tld != allowed_domain:
            continue

        if "Authenticate" in url:
            continue

        # if we've already visited that page, don't add it to queue
        if url in link_set:
            continue

        # otherwise, this link is good and should be added to queue for future handling
        queue.append(url)

        # add url to set so we don't revisit it
        link_set.add(url)
    # end of for loop that adds handles all links on page 


    # TRAIN MODEL WITH TEXT

    # extract text from current web page
    for script in soup(["script", "style"]):
        script.extract()

    # Get all the text out of the soup
    text = soup.get_text()

    # Break into lines and remove leading and trailing space on each line
    lines = (line.strip() for line in text.splitlines())

    # Break multi-headlines into a line each
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))

    # Drop blank lines and join everything into a single string
    textMerg = '\n'.join(chunk for chunk in chunks if chunk)

    try: 
        # write text to the temp txt file
        with open(r"\\VAPP-SRV-FS3\axc22$\Projects\Chat_bot\temp.txt", 'w+', encoding='utf-8') as f:
            # clear anything that was in temp.txt from last iteration
            f.truncate(0)
            
            # write new scraped wesbite to .txt file
            f.write(textMerg)

            f.flush()
            os.fsync(f.fileno())

            # load and reformat text to be ready for training
            loader = TextLoader(r"\\VAPP-SRV-FS3\axc22$\Projects\Chat_bot\temp.txt", encoding='utf-8')
            documents = loader.load()

            text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            docs = text_splitter.split_documents(documents)

            # add recent website information to data base
            db.add_documents(docs)

    except:
        db.save_local('U:\Projects\Chat_bot')

    # give the computer a second to close the file
    # and time to breath
    time.sleep(8.0)
# end of function
        




# need to add first link (starting point) to queue so loop starts
queue.append("https://www.cooperative.com/Pages/default.aspx")
link_set.add("https://www.cooperative.com/Pages/default.aspx") 

count = 0

# while there are links to visit, keep scraping and training
while queue:
    # pop first in link and set it to url
    url = queue.popleft()
    
    process_link(url)
    
    count += 1
    print(count)
    print(url + '\n')


# once we have scraped entire website and filled data base
# we should save the now expanded data base
db.save_local('U:\Projects\Chat_bot')

















# 'https://www.cooperative.com/_trust/default.aspx?trust=SP2016_PING_PROD&ReturnUrl=/_layouts/15/Authenticate.aspx?Source=%2fPages%2fdefault.aspx'

# 'https://www.cooperative.com/topics/advocacy'














# also, test its behavior with pdfs

# look for Pages/Secure to show that it doesn't break the fire wall

# each time you process a link, you scrape all links on that url's page
# I have a counter in the while loop to tell me how many times
# the process_link function is called, and it only gets called once
# before there is an error, so there is something wrong with calling
# process_link with the top priority link in the queue
# string must be formatted differently when scraped
# breakpoint to find out
        



# NOTES FOR WHEN I CONTINUE WORKING ON IT: WHERE I LEFT OF DEBUGGING

# on the second iteration of the while loop, this link:
    # "https://www.cooperative.com/topics/advocacy/Pages/Political-Advocacy-Overview-and-Key-Contacts/"
    # which can be clearly found on the page:
    # https://www.cooperative.com/topics/advocacy/Pages/default.aspx
        # this is the page that the second iteration of the while loop runs on (the first page after the home page that the function scrapes)
# 260 links are scraped, but somehow, all of them are either not proper links or already in link_set
# the link pasted above (Political-Advocacy...) is also apparanetly not in links (the list of links found when scraping the page)

# I have a feeling the issue is becuase some links the way they are captured
# by beautiful soup, do not have the .aspx in their ending\


# next steps for debugging this is to compare more detailed counts 
# 170 urls in the first iteration are valid and added to the link_set
# in the next iteration, 170 elements reach the stage right before checking
    # whether or not the url is already in the set; none pass this condition 
# I have a feeling the issue is that the scraper is somehow scraping the home page again
# or at least somehow scraping the exact same set of links
# even though the amount of elements in "links" list is different in the second iteration
# a lot of those links (<a> tags) are just buttons 
# again, the amount of actual links that are in the list in the second iteration
# is the exact same as the first

# another possibility is that the first iteration scrapes more than I think it scrapes 
    # (somehow goes beyond its own page)
# but I don't think this is the case

# i think the error is in the scraper in that it doesn't scrape the new url, it just rescrapes the old one

# error is that the scraper is not properly scraping. Not sure why