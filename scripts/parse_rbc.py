import time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
from urllib.parse import urljoin

from io import BytesIO
from urllib.request import urlopen

import json
import hashlib
import os
import glob
import re


import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

import pandas as pd
from sklearn.utils import shuffle



def parse(link):
    headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'}
    session = requests.Session()
    retry = Retry(connect=3, backoff_factor=0.5)
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    session.get(link, headers=headers)
    response = session.get(link, headers=headers)
    soup = BeautifulSoup(response.text, 'html.parser')
    return soup

def write_To_json(file_name, entry):
    try :
        with open(file_name) as json_file:
            data = json.load(json_file)
    except :
        data = {}

    data.update(entry)

    with open(file_name, "w+") as write_file:
        json.dump(data, write_file)

def index_with_hash(dict_entry):
    try:
        return {hashlib.md5(dict_entry['article_text'].encode()).hexdigest(): dict_entry}
    except Exception as e:
        print(e, dict_entry)


def parse_page(category, url, file_name):

    options = Options()
    options.add_argument("start-maximized")
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

    driver.get(url)
    time.sleep(2)  # Allow 2 seconds for the web page to open
    scroll_pause_time = 1 # You can set your own pause time. My laptop is a bit slow so I use 1 sec
    screen_height = driver.execute_script("return window.screen.height;")   # get the screen height of the web
    i = 1

    while True:
        # scroll one screen height each time
        driver.execute_script("window.scrollTo(0, {screen_height}*{i});".format(screen_height=screen_height, i=i))  
        i += 1
        time.sleep(scroll_pause_time)
        # update scroll height each time after scrolled, as the scroll height can change after we scrolled the page
        scroll_height = driver.execute_script("return document.body.scrollHeight;")  
        # Break the loop when the height we need to scroll to is larger than the total scroll height
        if (screen_height) * i > scroll_height:
            break 

        
    soup = BeautifulSoup(driver.page_source, "html.parser")
    articles = {}

    for element in soup.body.findAll('div', attrs={'class': 'item item_image-mob js-category-item'}):
        articles['url'] = element.find('a').get('href')
        articles['date'] = element.select_one('span.item__category').text.strip()
        articles['title'] = element.select_one('span.rm-cm-item-text').text.strip()
        articles['category'] = category

        soup = parse(articles['url'])
        corpus = {}
        trash = ['article__main-image', 'article__header__info-block', 'pro-anons', 'article__authors', 'social-networks__content', 'article__inline-item']

        for text_div in soup.findAll('div', attrs={'class':'l-col-main'}):
            extracted_trash = [t.extract() for trash_class in trash for t in text_div.findAll('div', attrs={'class':trash_class})]
            del extracted_trash

            for tag_bar in text_div.findAll('div', attrs={'class':'article__tags__container'}):
                tags = [tag.string for tag in tag_bar.findAll('a') if tag_bar is not None ]

            headline = text_div.find('h1')
            overview = text_div.find('div', attrs = {'article__text__overview'})
            if text_div is not None and overview is not None and headline is not None and tags is not None:
                corpus['headline'] = str(headline.text).strip()
                corpus['article_overview'] = overview.span.text
                corpus['category'] = category
                corpus['tags'] = tags
                tag_bar.extract()
                corpus['article_text'] = (re.sub(r'\s+', ' ', text_div.text)).str.replace('Авторы Теги', '')
                
            else:
                continue
        if bool(corpus):        
            entry = index_with_hash(corpus)
        try :
            with open(file_name) as json_file:
                data = json.load(json_file)
        except :
            data = {}
            
        data.update(entry)

        with open(file_name, "w+") as write_file:
            json.dump(data, write_file)


def main(dataset = '../dataset/rbc_dataset.json'):
    link_to_parse = {}
    link_to_parse['politics'] = 'https://www.rbc.ru/politics/?utm_source=topline'
    link_to_parse['economics'] = 'https://www.rbc.ru/economics/?utm_source=topline'
    link_to_parse['society'] = 'https://www.rbc.ru/society/?utm_source=topline'
    link_to_parse['business'] = 'https://www.rbc.ru/business/?utm_source=topline'
    link_to_parse['tech'] = 'https://www.rbc.ru/technology_and_media/?utm_source=topline'
    link_to_parse['finance'] = 'https://www.rbc.ru/finances/?utm_source=topline'

    file_name = dataset
    for category, link in link_to_parse.items():
        parse_page(category, link, file_name)



def shuffle_dataset(file_name='../dataset/hashed_rbc_dataset.json', shuffled_dataset_file='../dataset/shuffled_rbc_dataset.json'):
    df = pd.read_json(file_name)
    df = df.transpose()
    df = shuffle(df)
    df.to_json(shuffled_dataset_file, orient='index')


def split_to_textfiles(dataset_json='../dataset/shuffled_rbc_dataset.json', headlines_txt_path='../dataset/rbc_only_headlines.txt', annotation_txt_path='../dataset/rbc_only_annotation.txt', article_body_txt_path='../dataset/rbc_only_body.txt'):
    df = pd.read_json(dataset_json)
    df = df.transpose()
    special_separator = '<@$>\n\n'
    with open(headlines_txt_path, 'w', encoding = 'utf-8') as h, open(annotation_txt_path, 'w', encoding = 'utf-8') as a, open(article_body_txt_path, 'w', encoding = 'utf-8') as b:
        for rec_index, rec in df.iterrows():
            h.write(rec['headline'] + special_separator)
            a.write(rec['article_overview'] + special_separator)
            b.write(rec['article_text'] + special_separator)