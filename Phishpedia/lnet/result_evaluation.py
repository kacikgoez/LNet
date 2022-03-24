import sqlite3
import pickle
import json
import os
from src.siamese_pedia.utils import brand_converter
import tldextract

with open("../src/siamese_pedia/domain_map.pkl", "rb") as domainMap:
    domain_map = pickle.load(domainMap)

domain_map['Verizon Wireless'].append('verizon')
domain_map['Google'].append('blog')
domain_map['eBay'].append('ebay-kleinanzeigen')
domain_map['blizzard'].append('battlenet')
domain_map['Barclays Bank Plc'].append('barclaycardus')

exampleFile = r"/Volumes/GoogleDrive/Meine Ablage/LNet/Phishpedia/lnet/test.txt"
path = r"/Volumes/GoogleDrive/Meine Ablage/holle2/data.sqlite3"

con = sqlite3.connect(path)
cur = con.cursor()

cur.execute("""SELECT capture_id, details, url FROM (SELECT capture_id, details, suspect_url_id FROM capture JOIN 
    url_source USING(suspect_url_id) WHERE details != 'null' ORDER BY capture_id) 
    JOIN suspect_url USING(suspect_url_id)""")
dbItems = cur.fetchall()
hashItems = {}
t = 0
totallist = []

counter = {}
for dbItem in dbItems:
    json_array = json.loads(dbItem[1])
    if 'target' in json_array:
        target = str(json_array['target']).casefold().strip().replace("\"","").replace("&amp;", "&")
    else:
        if 'timestamp' in json_array:
            target = "none"
        elif 'tranco_rank' in json_array:
            target = "none"
        elif 'level' in json_array:
            target = "none"
        else:
            target = str(json_array)
    if target == 'interactivecorp':
        target = 'ourtime dating'
    elif target == 'office365':
        target = 'microsoft'
    elif target == 'outlook':
        target = 'microsoft'
    elif target == 'generic/spear phishing':
        target = 'none'
    elif target == 'internal revenue service':
        target = 'irs'
    elif target == 'tsb':
        target = 'tsb bank limited'
    elif target == "\"wells fargo\"":
        target = "wells fargo & company"
    elif target == "global sources (hk)":
        target = "global sources"
    elif target == "dgi french tax authority dgi (french tax authority)":
        target = "dgi french tax authority"
    elif target == "paypal inc.":
        target = "paypal"
    elif target == "hsbc group":
        target = "hsbc bank"
    elif target == "yahoo":
        target = "yahoo!"
    elif target == "caixa":
        target = "caixa economica federal"
    elif target == "absa bank":
        target = "absa group"
    elif target == "three":
        target = "three uk"
    elif target == "free (isp)":
        target = "free isp"#
    elif target == "halifax":
        target = "halifax bank of scotland"
    elif target == "natwest bank":
        target = "natwest personal banking"
    elif target == "dhl":
        target = "dhl airways"
    elif target == "facebook, inc.":
        target = "facebook"
    if int(dbItem[0]) not in hashItems or hashItems[int(dbItem[0])] == 'none':
        if str(target.casefold().strip()) not in totallist:
            totallist.append(str(target.casefold().strip()))
        hashItems[int(dbItem[0])] = [target.casefold().strip(), str(dbItem[2])]
        if target.casefold().strip() not in counter:
            counter[target.casefold().strip()] = 0
        else:
            counter[target.casefold().strip()] += 1

print(sorted(counter.items(), key=lambda x: x[1], reverse=True))

totalTrues = 0

newTruePosCount = 0
newFalsePosCount = 0
newTrueNegCount = 0
newFalseNegCount = 0

newCount = 0
FalseCount = 0

pediaTruePosCount = 0
pediaFalsePosCount = 0
pediaTrueNegCount = 0
pediaFalseNegCount = 0

sameCount = 0

with open(exampleFile, "r") as fdes:
    lines = fdes.readlines()
    print("Loading database, please wait ...")
    c = 0
    shareC = 0
    pedia = 0
    for listItem in lines:
        listItem = listItem.split("\t")
        currentItem = listItem[0][:8]
        matchItem = listItem[1].casefold().strip()
        if matchItem == '0':
            print(listItem)
        pediaItem = listItem[4].casefold().strip()
        if currentItem.isnumeric():
            if int(currentItem) in hashItems:
                target = hashItems[int(currentItem)]
                if pediaItem != 'none':
                    pedia += 1

                #if str(matchItem) in str(target) and pediaItem != matchItem and matchItem != 'none' and matchItem != 'cox communications' and str(target) != 'none':
                #if str(matchItem) not in target.casefold().strip() and (target != 'none' or 'generic' not in target):
                if (pediaItem not in target[0]) and matchItem == pediaItem and target[0] == 'none':
                    print(currentItem + "-screenshot.png", matchItem, pediaItem, target[0])
                    c += 1
                    print(c)
                    pass
                #if matchItem != pediaItem and pediaItem == 'none':
                    #print(brand_converter(matchItem))
                    #if brand_converter(matchItem) != 'none':
                '''match = False
                currentBrand = False
                for i in domain_map.items():
                    if str(tldextract.extract(target[1]).domain).casefold() in i[1]:
                        #print(tldextract.extract(target[1]).domain, i)
                        #print(str(currentItem) + "-screenshot.png", target[1])
                        totalTrues += 1
                        currentBrand = i[0].casefold()
                        match = True

                if currentBrand:
                    if brand_converter(matchItem).casefold() != currentBrand and brand_converter(pediaItem).casefold() == currentBrand:
                        pass

                    if brand_converter(matchItem).casefold() == currentBrand:
                        newTruePosCount += 1
                    else:
                        newFalseNegCount += 1

                    if brand_converter(pediaItem).casefold() == currentBrand:
                        pediaTruePosCount += 1
                    else:
                        pediaFalseNegCount += 1
                        #print("PEDIA https://drive.google.com/drive/u/0/search?q=" + str(currentItem) + "-screenshot.png", pediaItem, target[1])

                if not currentBrand:
                    if brand_converter(matchItem).casefold() != 'none':
                        newFalsePosCount += 1
                        #print("NEW https://drive.google.com/drive/u/0/search?q=" + str(currentItem) + "-screenshot.png", matchItem, target[1])
                    else:
                        newTrueNegCount += 1

                    if brand_converter(matchItem).casefold() != 'none' and brand_converter(pediaItem).casefold() == 'none':
                        #print("SHARED https://drive.google.com/drive/u/0/search?q=" + str(currentItem) + "-screenshot.png", matchItem, target[1])
                        shareC += 1

                    if brand_converter(pediaItem).casefold() != 'none':
                        pediaFalsePosCount += 1
                        #print("PEDIA https://drive.google.com/drive/u/0/search?q=" + str(currentItem) + "-screenshot.png", pediaItem, target[1])
                    else:
                        pediaTrueNegCount += 1'''

    '''print(totalTrues)
    print(c)
    print(c/pedia)
    print(c/len(lines))
    print(len(lines))'''
    print("LNET: ---------------")
    print("True Positives:" + str(newTruePosCount))
    print("False Positives:" + str(newFalsePosCount))
    print("True Negatives:" + str(newTrueNegCount))
    print("False Negatives:" + str(newFalseNegCount))

    print("PEDIA: ---------------")
    print("True Positives:" + str(pediaTruePosCount))
    print("False Positives:" + str(pediaFalsePosCount))
    print("True Negatives:" + str(pediaTrueNegCount))
    print("False Negatives:" + str(pediaFalseNegCount))

    print("SHARED", shareC)