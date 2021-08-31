import os
import difflib
import pdfplumber
import pandas as pd
from operator import itemgetter
from itertools import groupby


def getFileNameInDirectory(directoryPath):
    fileNameList = os.listdir(directoryPath)
    return fileNameList


def extractPDFContent(filePath):
    pdf = pdfplumber.open(filePath)
    page = pdf.pages[0]
    text = page.extract_text(x_tolerance=3)
    return text


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
