import wikipedia
import csv
import io
import re

wikipedia.set_lang("es")

subjects = [["Ubuntu","Tecnología"],
            ["arte","pintura"],
            ["política","parlamento","leyes"]]

def stringtocsv(text,ident,csvFile="data\classificator.csv"):
    with io.open(csvFile, 'a', encoding="utf-8") as File:
        writer = csv.writer(File, delimiter=",", quoting=csv.QUOTE_MINIMAL,lineterminator='\n')
        writer.writerow([text,ident])

for i in range(0,len(subjects)):
    subject = subjects[i]
    for j in range(0, len(subject)):
        #data = wikipedia.summary(subject[j])
        data = wikipedia.page(subject).content
        data = ' '.join(re.sub("(@[A-Za-z0-9]+)|(#[A-Za-z0-9]+)", " ", data).split())
        data = ' '.join(re.sub("\[[ ^\(]+[:digit:]+[^ \(]+\]", "",data).split())
        data = ' '.join(re.sub("(\w+:\/\/\S+)", " ", data).split())
        data = ' '.join(re.sub("[\.\,\!\?\:\;\-\=\[\]\(\)\/\»\«\"\'\‘\’\“\”\%\—]", "", data).split())
        data = data.lower()
        x=140
        last = 0
        contador = 0
        splitData = data.split()
        #subData = [data[y - x:y] for y in range(x, len(data) + x, x)]
        for z in range(0,len(splitData)):
            contador += len(splitData[z]) + 1
            if contador > x:
                finalSent = splitData[last:z-1]
                stringtocsv(' '.join(finalSent), i)
                contador = len(splitData[z])
                last = z

