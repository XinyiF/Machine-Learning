from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from numpy import *

class fileDialogdemo(QWidget):
    def loadDataSet(self):
        postingList = [['my', 'dog', 'has', 'flea', 'problems', 'cats', 'please'],  # talk about pets
                       ['clearance', 'shipped', 'Ultimate', 'Order', 'Buy', 'Cheap', ' mg', 'discount'],
                       # Words from commerce spam email
                       ['hotel', 'trip', 'is', 'so', 'cute', 'scenic', 'love', 'pick'],  # talk about trip
                       ['100%', 'gains ', 'earn', '$', 'OFF', 'price'],
                       # Words from financial for employment  spam email
                       ['Hi', 'I\'m', 'they', 'my', 'dinner', 'how', 'to', 'guy', 'pictures'],  # email from friends
                       ['Click', 'member', 'Membership', 'Open', 'Unsubscribe', 'Notspam'],  # I always meet them in ads
                       ['yesterday', 'train', 'talked', 'to', 'party', 'commented'],
                       ['local', 'doorstep', 'Community', 'deliver', 'seafood', 'chance']]
        # if you change the number of row of thesaurus, you should also modify this array
        classVec = [0, 1, 0, 1, 0, 1, 0, 1]  # 1 is spam, 0 not
        return postingList, classVec

    def createVocabList(self,dataSet):
        vocabList = []
        for i in range(len(dataSet)):
            for j in range(len(dataSet[i])):
                if not dataSet[i][j] in vocabList:
                    vocabList.append(dataSet[i][j])
        return vocabList

    # if the word in the vocabList
    def setOfWords2Vec(self,vocabList, inputSet):
        returnVec = zeros(len(vocabList))
        for word in inputSet:
            if word in vocabList:
                returnVec[vocabList.index(word)] = 1
        return returnVec

    # each col of trainMatrix means one word
    def trainNB0(self,trainMatrix, listClasses):
        pAbusive = float(sum(listClasses)) / float(len(listClasses))
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
        p1Vect = p1Num / p1Denom
        p0Vect = p0Num / p0Denom
        return p0Vect, p1Vect, pAbusive

    def classifyNB(self,vec2Classify, p0Vec, p1Vec, pAbusive):
        # only count for words in the list
        p1 = sum(vec2Classify * p1Vec) + pAbusive
        p0 = sum(vec2Classify * p0Vec) + 1.0 - pAbusive
        if p1 > p0:
            return 1
        else:
            return 0


    def readMail(self,filepath):
        content = []
        file_object = open(filepath)
        try:
            for line in file_object:
                line = line.split()
                for i in line:
                    content.append(i)
        finally:
            file_object.close()
        return content


    def __init__(self,parent=None):
        super(fileDialogdemo, self).__init__(parent)
        layout=QVBoxLayout()

        self.le = QLabel('')
        layout.addWidget(self.le)

        self.btn1 = QPushButton('Load mail')
        self.btn1.clicked.connect(self.getFiles)
        layout.addWidget(self.btn1)

        self.contents = QTextEdit()
        layout.addWidget(self.contents)

        self.setLayout(layout)
        self.setWindowTitle('Sample')

    def getFiles(self):

        dig = QFileDialog()

        dig.setFileMode(QFileDialog.AnyFile)

        dig.setFilter(QDir.Files)

        if dig.exec_():
            filenames = dig.selectedFiles()

            testMail = self.readMail(filenames[0])
            listOPosts, listClasses = self.loadDataSet()
            myVocabList = self.createVocabList(listOPosts)  # delete repeat words
            trainMat = []
            for i in range(0, len(listOPosts)):
                trainMat.append(self.setOfWords2Vec(myVocabList, listOPosts[i]))
            p0V, p1V, pAb = self.trainNB0(trainMat, listClasses)
            thisDoc = array(self.setOfWords2Vec(myVocabList, testMail))
            if self.classifyNB(thisDoc, p0V, p1V, pAb) == 0:
                data='classified as ham mail'
            else:
                data='classified as spam mail'
            self.contents.setText(data)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = fileDialogdemo()
    ex.show()
    sys.exit(app.exec_())


