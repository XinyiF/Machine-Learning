from numpy import *
import email
import imaplib

def loadDataSet():
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'cats', 'please'], # talk about pets
                   ['clearance', 'shipped', 'Ultimate', 'Order', 'Buy', 'Cheap', ' mg', 'discount'], # Words from commerce spam email
                   ['hotel', 'trip', 'is', 'so', 'cute', 'scenic', 'love', 'pick'], # talk about trip
                   ['100%', 'gains ', 'earn', '$', 'OFF','price'], # Words from financial for employment  spam email
                   ['Hi', 'I\'m', 'they', 'my', 'dinner', 'how', 'to', 'guy', 'pictures'], # email from friends
                   ['Click', 'member', 'Membership', 'Open', 'Unsubscribe', 'Notspam'],# I always meet them in ads
                   ['yesterday','train','talked','to','party','commented'],
                   ['local','doorstep','Community','deliver','seafood','chance']]
    #if you change the number of row of thesaurus, you should also modify this array
    classVec = [0, 1, 0, 1, 0, 1,0,1]  # 1 is spam, 0 not
    return postingList, classVec


def createVocabList(dataSet):
    vocabList=[]
    for i in range(len(dataSet)):
        for j in range(len(dataSet[i])):
            if not dataSet[i][j] in vocabList:
                vocabList.append(dataSet[i][j])
    return vocabList

#if the word in the vocabList
def setOfWords2Vec(vocabList, inputSet):
    returnVec = zeros(len(vocabList))
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
    return returnVec

#each col of trainMatrix means one word
def trainNB0(trainMatrix,listClasses):
    pAbusive = float(sum(listClasses))/float(len(listClasses))
    p0Num = zeros(len(trainMatrix[0]))
    p1Num = zeros(len(trainMatrix[0]))
    p0Denom = 0
    p1Denom = 0
    for i in range(len(trainMatrix)):
        if listClasses[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1Vect = p1Num/p1Denom
    p0Vect = p0Num/p0Denom
    return p0Vect,p1Vect,pAbusive

def classifyNB(vec2Classify, p0Vec, p1Vec, pAbusive):
    #only count for words in the list
    p1 = sum(vec2Classify * p1Vec) + pAbusive
    p0 = sum(vec2Classify * p0Vec) + 1.0 - pAbusive
    if p1>p0:
        return 1
    else:
        return 0

def fetchRecentMail(username,password):
    mail = imaplib.IMAP4_SSL("imap.gmail.com")
    mail.login(username, password)

    mail.select("inbox")
    mail.list()
    result, data = mail.search(None, 'ALL')
    inbox_item_list = data[0].split()
    most_recent = inbox_item_list[5]
    result2, email_data = mail.uid('fetch', most_recent, '(RFC822)')
    raw_email = email_data[0][1].decode()
    email_message = email.message_from_string(raw_email)
    print(email_message['From'])
    return email_message['subject'].split()


def readMail(filename):
    content=[]
    file_object = open(filename)
    try:
        for line in file_object:
            line=line.split()
            for i in line:
                content.append(i)
    finally:
        file_object.close()
    return content


# username='address'
# password='password'
# testMail=fetchRecentMail(username,password)

#calssification
testMail = readMail('/Users/user/Desktop/532Project/email/ham/7.txt')
listOPosts, listClasses = loadDataSet()
myVocabList = createVocabList(listOPosts)  # delete repeat words
trainMat = []
for i in range(0, len(listOPosts)):
    trainMat.append(setOfWords2Vec(myVocabList, listOPosts[i]))
p0V, p1V, pAb = trainNB0(trainMat, listClasses)
thisDoc = array(setOfWords2Vec(myVocabList, testMail))
if classifyNB(thisDoc, p0V, p1V, pAb)==0:
    print('classified as ham mail')
else:
    print('classified as spam mail')

#error rate
# error=0
# for j in range(1,17):
#     testMail = readMail('/Users/user/Desktop/532Project/email/spam/' + str(j) + '.txt')
#     listOPosts, listClasses = loadDataSet()
#     myVocabList = createVocabList(listOPosts)  # delete repeat words
#     trainMat = []
#     for i in range(0, len(listOPosts)):
#         trainMat.append(setOfWords2Vec(myVocabList, listOPosts[i]))
#
#     p0V, p1V, pAb = trainNB0(trainMat, listClasses)
#
#     thisDoc = array(setOfWords2Vec(myVocabList, testMail))
#     # print('classified as', classifyNB(thisDoc, p0V, p1V, pAb))
#     if classifyNB(thisDoc, p0V, p1V, pAb)==0:
#         error+=1
#         #you can print to find misclassified mail
#         #print(j)
# print(float(error/16))






