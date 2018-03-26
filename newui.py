from PyQt5 import QtGui,QtCore
import sys
import mainwindow
import numpy as np
import pylab
import time
import pyqtgraph
import pullCrypto
from collections import defaultdict

class ExampleApp(QtGui.QMainWindow, mainwindow.Ui_MainWindow):
    def __init__(self, parent=None):
        pyqtgraph.setConfigOption('background', 'k') #before loading widget
        super(ExampleApp, self).__init__(parent)
        self.setupUi(self)
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update)
        self.graphicsView.plotItem.showGrid(True, True, 0.7)
        self.dataObj = pullCrypto.cryptoWrapper();
        cols = list(self.dataObj.df.columns.values)
        markPairSplit = [x.split(":") for x in cols]
        self.markDict = defaultdict(list)
        for i in markPairSplit[2:]:
            self.markDict[i[0]].append(i[1])
        self.marketCombo.addItems(list(self.markDict.keys()))
        self.marketCombo.currentIndexChanged.connect(self.on_marketComboChange)
        self.lookupButton.clicked.connect(self.on_bttn_pressed)
        self.currPair = "bitfinex:btcusd"
        self.interval = 0
        self.timer.start(2500)
        

    def on_marketComboChange(self, value):
        self.pairCombo.clear()
        self.pairCombo.addItems(self.markDict[self.marketCombo.currentText()])

    def on_bttn_pressed(self):
        mark = self.marketCombo.currentText()
        mark += ":" + self.pairCombo.currentText()
        self.currPair = mark
        self.pricePairLabel.setText(mark)
        
        
    def update(self):
        self.dataObj.update()
        if (self.interval%5==0):
            self.dataObj.save()
        self.interval +=1
        C=pyqtgraph.mkColor('g')
        pen=pyqtgraph.mkPen(color=C,width=1)
        dates = self.dataObj.df.Date.values.tolist()
        self.pricePairVal.setText("%4f" % self.dataObj.df[self.currPair].iloc[-1])
        self.graphicsView.plot(dates,self.dataObj.df[self.currPair].values.tolist(),pen=pen,clear=True)
        

if __name__=="__main__":
    app = QtGui.QApplication(sys.argv)
    form = ExampleApp()
    form.show()
    form.update() #start with something
    app.exec_()
    print("DONE")
