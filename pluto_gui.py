from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow, QInputDialog

import cv2
import numpy as np

import pluto as pl

util = pl.PlutoObject(None)

class Ui_MainWindow(QMainWindow):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(547, 349)
        self.setWindowIcon(QtGui.QIcon("D:\\Codeing\\Pluto-Nightly\\icon.png"))
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        
        self.selectImageButton = QtWidgets.QPushButton(self.centralwidget)
        self.selectImageButton.setGeometry(QtCore.QRect(20, 10, 111, 31))
        self.selectImageButton.setObjectName("selectImageButton")
        self.selectImageButton.clicked.connect(self.selectpath)
        
        self.imgPathLabel = QtWidgets.QLabel(self.centralwidget)
        self.imgPathLabel.setGeometry(QtCore.QRect(20, 50, 681, 21))
        self.imgPathLabel.setObjectName("imgPathLabel")
        
        self.analyseButton = QtWidgets.QPushButton(self.centralwidget)
        self.analyseButton.setGeometry(QtCore.QRect(20, 90, 75, 23))
        self.analyseButton.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.analyseButton.setObjectName("analyseButton")
        self.analyseButton.clicked.connect(self.do_foxnews)
        
        self.analyseButton2 = QtWidgets.QPushButton(self.centralwidget)
        self.analyseButton2.setGeometry(QtCore.QRect(100, 90, 75, 23))
        self.analyseButton2.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.analyseButton2.setObjectName("analyseButton2")
        self.analyseButton2.clicked.connect(self.do_nyt)
        
        self.analyseButton3 = QtWidgets.QPushButton(self.centralwidget)
        self.analyseButton3.setGeometry(QtCore.QRect(180, 90, 75, 23))
        self.analyseButton3.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.analyseButton3.setObjectName("analyseButton3")
        self.analyseButton3.clicked.connect(self.do_facebook)
        
        self.analyseButton4 = QtWidgets.QPushButton(self.centralwidget)
        self.analyseButton4.setGeometry(QtCore.QRect(260, 90, 75, 23))
        self.analyseButton4.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.analyseButton4.setObjectName("analyseButton3")
        self.analyseButton4.clicked.connect(self.do_twitter)
        
        self.analyseSep = QtWidgets.QFrame(self.centralwidget)
        self.analyseSep.setGeometry(QtCore.QRect(20, 70, 201, 16))
        self.analyseSep.setFrameShadow(QtWidgets.QFrame.Plain)
        self.analyseSep.setLineWidth(1)
        self.analyseSep.setFrameShape(QtWidgets.QFrame.HLine)
        self.analyseSep.setObjectName("analyseSep")
        
        self.resultSep = QtWidgets.QFrame(self.centralwidget)
        self.resultSep.setGeometry(QtCore.QRect(20, 120, 201, 16))
        self.resultSep.setFrameShadow(QtWidgets.QFrame.Plain)
        self.resultSep.setLineWidth(1)
        self.resultSep.setFrameShape(QtWidgets.QFrame.HLine)
        self.resultSep.setObjectName("resultSep")
        
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(20, 140, 211, 161))
        self.label.setScaledContents(False)
        self.label.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignTop)
        self.label.setObjectName("label")
        self.label.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)

        MainWindow.setCentralWidget(self.centralwidget)
        
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 547, 22))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Pluto"))
        self.selectImageButton.setText(_translate("MainWindow", "Select Image..."))
        self.imgPathLabel.setText(_translate("MainWindow", "Selected Image: [null]"))
        self.analyseButton.setText(_translate("MainWindow", "Fox News"))
        self.analyseButton2.setText(_translate("MainWindow", "NYT"))
        self.analyseButton3.setText(_translate("MainWindow", "FB"))
        self.analyseButton4.setText(_translate("MainWindow", "Twitter"))
        self.label.setText(_translate("MainWindow", "Result:"))
    
    def selectpath(self):
        userInput, okPressed = QInputDialog.getText(self, "Input Image Path", "Please enter the path to your image:")
        if okPressed:
            self.imgPathLabel.setText("Selected Image: " + userInput)
            # self.photo.setPixmap(QtGui.QPixmap(userInput))
            self.img_path = userInput
        self.update()
    
    def do_foxnews(self):
        # print("Analysed pressed!")
        # print(self.img_path)
        try:
            img = img = pl.read_image(self.img_path)
            author, subtitle, headline, pubsplit, dotsplit, msg = fox_analyse(img)
            self.label.setText("Result: \nAuthor: " + author + "\nSubtitle: " + subtitle + "\nHeadline: " + headline + "\nPlublished: " + pubsplit + "\nTopic: " + dotsplit)
        except Exception as e:
            self.label.setText("Error: \n" + str(e))
            print(e)
        self.update()
    
    def do_facebook(self):
        # print("Analysed pressed!")
        # print(self.img_path)
        try:
            img = img = pl.read_image(self.img_path)
            author, date, body_text, engagement_text = pl.Facebook(img).analyse()
            self.label.setText("Result: \nAuthor: " + author + "\nPublished: " + date + "\nPost: " + body_text + "\nEngagement: " + engagement_text)
        except Exception as e:
            self.label.setText("Error: \n" + str(e))
            print(e)
        self.update()
    
    def do_twitter(self):
        # print("Analysed pressed!")
        # print(self.img_path)
        try:
            img = img = pl.read_image(self.img_path)
            name, handle, text = pl.Twitter(img).analyse()
            self.label.setText("Result: \nName: " + name + "\nHandle: " + handle + "\nText: " + text)
        except Exception as e:
            self.label.setText("Error: \n" + str(e))
            print(e)
        self.update()
    
    def do_nyt(self):
        # print("Analysed pressed!")
        # print(self.img_path)
        try:
            img = img = pl.read_image(self.img_path)
            nyt = pl.NYT(img)
            headline, subtitle = nyt.analyse()
            self.label.setText("Result: \nHeadline: " + headline + "\nSubtitle: " + subtitle)
        except Exception as e:
            self.label.setText("Error: \n" + str(e))
            print(e)
        self.update()

    def update(self):
        self.centralwidget.adjustSize()
        self.imgPathLabel.adjustSize()
        self.label.adjustSize()

def fox_analyse(img, display=False):
    msg = ""
    og_shape = img.shape
    og_img = img.copy()
    img = cv2.resize(img, (512, 512))
    if display: pl.show_image(img)
    black = np.zeros((512, 512))
    
    for i in range(len(black)):
        for j in range(len(black[0])):
            temp = img[i][j]
            if (temp == [34, 34, 34]).all(): black[i][j] = 255
    blured = cv2.blur(black, (20, 20))

    for i in range(len(blured)):
        for j in range(len(blured[0])):
            if blured[i][j] < 40: blured[i][j] = 0
            else: blured[i][j] = 255

    msk = pl.expand_to_rows(blured)
    if display: pl.show_image(msk)

    og_size_msk = cv2.resize(msk, (og_shape[1], og_shape[0]))
    
    top = []
    heading = []
    bottom = []

    top_part = True
    bottom_part = False

    # print(img.shape)
    # cv2.imwrite("test1.jpg", og_img)
    for i in range(len(og_img)):
        if og_size_msk[i][0] > 1:
            heading.append(og_img[i])
            if top_part:
                top_part = False
                bottom_part = True
        elif top_part: top.append(og_img[i])
        else: bottom.append(og_img[i])

    heading = np.array(heading)
    bottom = np.array(bottom)
    top = np.array(top)
    
    # print(heading.shape, bottom.shape, top.shape)
    
    if display:
        pl.show_image(heading)
        pl.show_image(bottom)
        pl.show_image(top)

    try:
        ocr_result = util.ocr(heading)
        headline = util.ocr_cleanup(ocr_result)
    except Exception as e:
        ocr_result = "null"
        headline = "null"
        msg = "Error while performing ocr: ", str(e)

    cat_info_img = []
    top_len = len(top)
    for i in range(top_len, 0, -1):
        if top[i-1][0][0] > 250: cat_info_img.insert(0, top[i-1])
        else: break

    cat_info_img = np.array(cat_info_img)
    # print(cat_info_img.shape)
    if display: pl.show_image(cat_info_img)

    try:
        ocr_result = util.ocr(cat_info_img)
        clean_ocr = util.ocr_cleanup(ocr_result)
        # print("clean ocr", clean_ocr)
    except Exception as e:
        ocr_result = "null"
        clean_ocr = "null"
        msg = "Error while performing ocr: ", str(e)

    try:
        dotsplit = clean_ocr.split("-")[0][:-1].lstrip(" ")
        pubsplit = clean_ocr.split("Published")[1].lstrip(" ")
    except Exception as e:
        dotsplit = "null"
        pubsplit = "null"
        msg = "Error: " + str(e)
    
    subinfo_bottom = []

    stoper = False
    for row in bottom:
        subinfo_bottom.append(row)
        for pix in row:
            if pix[0] > 200 and pix[0] < 240 and pix[2] < 50 and pix[1] < 50:
                stoper = True
                break
        if stoper: break

    subinfo_bottom = np.array(subinfo_bottom[:-3])
    if display: pl.show_image(subinfo_bottom)
    try:
        subinfo = util.ocr_cleanup(util.ocr(subinfo_bottom))
    except Exception as e:
        subinfo = "null"
        msg = "Error while performing ocr: " + str(e)

    subsplit = subinfo.split()

    author_list = []
    subtitle_list = []
    subinfo_switcher = True

    for w in reversed(subsplit):
        if w == "By" and subinfo_switcher:
            subinfo_switcher = False
            continue
        if w == "News" or w == "Fox" or w == "|": continue
        if subinfo_switcher: author_list.insert(0, w)
        else: subtitle_list.insert(0, w)

    author = " ".join(author_list)
    subtitle = " ".join(subtitle_list)
    
    return author, subtitle, headline, pubsplit, dotsplit, msg

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())