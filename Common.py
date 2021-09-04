import os
import re
import difflib
import pdfplumber
import pandas as pd
from operator import itemgetter
from itertools import groupby
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import itertools
import nltk


from gensim.summarization import keywords
from gensim.summarization.summarizer import summarize

from nltk.stem.porter import PorterStemmer

porter_stemmer = PorterStemmer()
from nltk.stem.lancaster import LancasterStemmer

lancaster_stemmer = LancasterStemmer()
from nltk.stem import SnowballStemmer

snowball_stemmer = SnowballStemmer("english")
from nltk.stem import WordNetLemmatizer

wordnet_lemmatizer = WordNetLemmatizer()

from nltk.corpus import stopwords

stops = stopwords.words("english")


def getFileNameInDirectory(directoryPath):
    fileNameList = os.listdir(directoryPath)
    return fileNameList


def extractPDFContent(filePath):
    pdf = pdfplumber.open(filePath)
    page = pdf.pages[0]
    text = page.extract_text(x_tolerance=3)
    return text


def extractTXTContent(filePath):
    with open(filePath) as f:
        lines = f.readlines()
    text = "".join(lines)
    return text


def getMatchScore(jobDescriptionText, resumeText):
    cv = CountVectorizer()
    countMatrix = cv.fit_transform([jobDescriptionText, resumeText])
    matchPercentage = round(cosine_similarity(countMatrix)[0][1] * 100, 2)
    return matchPercentage


def groupbyFirstLetter(iteratorSeq):
    groupbyData = [
        list(words) for letter, words in groupby(sorted(iteratorSeq), key=itemgetter(0))
    ]
    return groupbyData


def getselectedAndUniqueKeywords(keywordDf):
    selectedKeywords = (
        pd.concat([keywordDf["snowball_stemmer"], keywordDf["wordnet_lemmatizer"]])
        .sort_values()
        .values.tolist()
    )
    selectedAndUniqueKeywords = set(selectedKeywords)
    return selectedAndUniqueKeywords


def filterSimilarityWords(inputList, thresholdRatio=0.7):
    li = []
    if len(inputList) > 0:
        li = [inputList[0]]
        for word in inputList:
            flag = 1
            for liWord in li:
                ratio = difflib.SequenceMatcher(None, word, liWord).ratio()
                if ratio > thresholdRatio:
                    flag = -1
                    break
            flag == 1 and li.append(word)
    return li


def getKeyWords(jobDescriptionText, keywordNum=10):
    keywordNum = 10
    tokens = nltk.wordpunct_tokenize(jobDescriptionText)
    tokenDf = pd.DataFrame(index=tokens)
    tokenDf["porter_stemmer"] = [porter_stemmer.stem(t) for t in tokens]
    tokenDf["lancaster_stemmer"] = [lancaster_stemmer.stem(t) for t in tokens]
    tokenDf["snowball_stemmer"] = [snowball_stemmer.stem(t) for t in tokens]
    tokenDf["wordnet_lemmatizer"] = [wordnet_lemmatizer.lemmatize(t) for t in tokens]
    idxs = list(tokenDf.columns)
    keywordDic = dict()
    for idx in idxs:
        tokensList = list(tokenDf[idx])
        text = " ".join(tokensList)
        keywordStr = keywords(text, ratio=0.3)
        keywordList = re.split("\n| ", keywordStr)
        keywordDic[idx] = keywordList[:keywordNum]
    keywordDf = pd.DataFrame.from_dict(keywordDic)
    keywordDf.index = keywords(jobDescriptionText, ratio=0.2).split("\n")[:keywordNum]
    selectedAndUniqueKeywords = getselectedAndUniqueKeywords(keywordDf)
    groupbyKeywords = groupbyFirstLetter(selectedAndUniqueKeywords)
    totalKeywords = [
        filterSimilarityWords(words, thresholdRatio=0.7) for words in groupbyKeywords
    ]
    keywordsList = list(itertools.chain(*totalKeywords))
    return keywordsList


def getSummarization(jobDescriptionText, ratio=0.1):
    text = summarize(jobDescriptionText, ratio=ratio)
    text = re.sub("\n", " ", text)
    return text


def generateDFtoHTML(df, index, htmlFileName):
    dfHTML = df.to_html(index=index)
    file = open(f"{htmlFileName}.html", "w")
    file.write(dfHTML)
    file.close()
