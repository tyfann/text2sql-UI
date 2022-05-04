# -*- ecoding: utf-8 -*-
# @ModuleName: main
# @Function: 
# @Author: Yufan-tyf
# @Time: 2022/5/4 15:34
import os, glob, shutil, sys, warnings, importlib, argparse
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QPlainTextEdit,
                             QWidget, QMessageBox, QLineEdit, QSpinBox)
from PyQt5 import uic


class mainWindow(QMainWindow):

    def __init__(self):
        super(mainWindow, self).__init__()
        uic.loadUi("ui/mainLayout.ui", self)

        self.generateButton.clicked.connect(self.generate)

    def generate(self):
        question = self.plainTextEdit_input.toPlainText()
        db_id = 'AI_SEARCH_' + str(self.spinBox_db.value())


    def closeEvent(self, e):
        reply = QMessageBox.question(self, '提示',
                                     "是否要关闭所有窗口?",
                                     QMessageBox.Yes | QMessageBox.No,
                                     QMessageBox.No)
        if reply == QMessageBox.Yes:
            e.accept()
            sys.exit(0)  # 退出程序
        else:
            e.ignore()


if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default='./models/pretrain1', help='enter the model path with model name.')
    args = parser.parse_args()

    app = QApplication(sys.argv)
    mainWin = mainWindow()

    mainWin.show()
    sys.exit(app.exec_())
