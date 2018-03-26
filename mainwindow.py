# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'mainwindow.ui'
#
# Created by: PyQt5 UI code generator 5.9
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1070, 696)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.graphicsView = PlotWidget(self.centralwidget)
        self.graphicsView.setGeometry(QtCore.QRect(25, 21, 761, 531))
        self.graphicsView.setObjectName("graphicsView")
        self.formLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.formLayoutWidget.setGeometry(QtCore.QRect(790, 20, 241, 341))
        self.formLayoutWidget.setObjectName("formLayoutWidget")
        self.formLayout = QtWidgets.QFormLayout(self.formLayoutWidget)
        self.formLayout.setContentsMargins(0, 0, 0, 0)
        self.formLayout.setObjectName("formLayout")
        self.pricePairLabel = QtWidgets.QLabel(self.formLayoutWidget)
        self.pricePairLabel.setObjectName("pricePairLabel")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.pricePairLabel)
        self.pricePairVal = QtWidgets.QLabel(self.formLayoutWidget)
        self.pricePairVal.setAutoFillBackground(False)
        self.pricePairVal.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.pricePairVal.setObjectName("pricePairVal")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.pricePairVal)
        self.marketCombo = QtWidgets.QComboBox(self.formLayoutWidget)
        self.marketCombo.setObjectName("marketCombo")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.SpanningRole, self.marketCombo)
        self.pairCombo = QtWidgets.QComboBox(self.formLayoutWidget)
        self.pairCombo.setObjectName("pairCombo")
        self.formLayout.setWidget(2, QtWidgets.QFormLayout.SpanningRole, self.pairCombo)
        self.lookupButton = QtWidgets.QPushButton(self.formLayoutWidget)
        self.lookupButton.setObjectName("lookupButton")
        self.formLayout.setWidget(3, QtWidgets.QFormLayout.SpanningRole, self.lookupButton)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1070, 22))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.pricePairLabel.setText(_translate("MainWindow", "price"))
        self.pricePairVal.setText(_translate("MainWindow", "TextLabel"))
        self.lookupButton.setText(_translate("MainWindow", "Look up market and pair"))

from pyqtgraph import PlotWidget
