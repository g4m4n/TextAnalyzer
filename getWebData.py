import re, os
import pickle
from collections import defaultdict
import requests
from bs4 import BeautifulSoup as bs4


def getPageContent(url):  # Función que devuelve todo el contenido HTML de una URL
    response = requests.get(url)
    return response.content


def html_to_text(html):
    soup = bs4(html, 'html.parser')
    return soup.get_text()


def getTextPage(url):  # Función para extraer todo el texto del contenido HTML de la pagina
    print("Aprendiendo de:", url)
    html = getPageContent(url)
    html_parsed = html_to_text(html)
    return html_parsed  # Este texto sera mediante el cual se entrenara al clasificador

def remove_urls(text):  # Metodo para eliminar urls de los textos
    return re.sub("https?://[^\s]+", "", text)


def getWords(text):  # Obtenemos todo el texto y lo dividimos por palabras
    assert type(text) in (str, bytes)
    text = remove_urls(text)
    words = re.findall("[a-z]{2,}", text, re.I)
    words = map(lambda x: x.lower(), words)
    return [word for word in words]


def wordFreq(text):  # Claculamos la frecuencia con la cual se repite una palabra
    words = getWords(text)
    count = defaultdict(int)
    for word in words:
        count[word] += 1
    return count


