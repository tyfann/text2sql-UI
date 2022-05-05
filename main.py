# -*- ecoding: utf-8 -*-
# @ModuleName: main
# @Function: 
# @Author: Yufan-tyf
# @Time: 2022/5/4 15:34
import json
import sys
import warnings

from PyQt5 import uic
from PyQt5.QtWidgets import (QApplication, QMainWindow, QMessageBox)

from schema_linking import build_cell_index, search_values
from text2sql import global_config
from text2sql.dataproc.dusql_dataset_v2 import load_tables
from text2sql_main import preprocess, inference, init_env, _set_proc_name, load_model


class mainWindow(QMainWindow):

    def __init__(self):
        super(mainWindow, self).__init__()
        uic.loadUi("ui/mainLayout.ui", self)

        self.generateButton.clicked.connect(self.generate)
        self.model = self.__load_model()
        db_schema = 'data/db_schema.json'
        db_content = 'data/db_content.json'
        self.dct_db, _ = load_tables(db_schema, db_content)
        build_cell_index(self.dct_db)

    @staticmethod
    def __load_model():
        config = global_config.gen_config(
            ['--mode', 'infer', '--init-model-param', 'model/model.pdparams', '--valid-set', 'data/preproc/valid.pkl',
             '--data-root', 'data/preproc'])
        init_env(config)
        model = load_model(config)
        return model

    def generate(self):

        question = self.plainTextEdit_input.toPlainText()
        db_id = 'AI_SEARCH_' + str(self.spinBox_db.value())

        lst_output = []
        question_id = f'qid{1:06d}'

        lst_output.append({
            "query": "",
            "question_id": question_id,
            "question": question,
            "db_id": db_id,
            "sql": "",
        })

        with open('data/valid.json', 'w') as f:
            f.write(json.dumps(lst_output, indent=2, ensure_ascii=False))
        self.schema_link(question, db_id)
        self.__preprocess()
        self.__inference()

    def schema_link(self, question, db_id):

        lst_output = []

        question_id = f'qid{1:06d}'
        db = self.dct_db[db_id]

        match_values = search_values(question, db)
        lst_output.append({
            "question_id": question_id,
            "question": question,
            "db_id": db_id,
            "match_values": match_values
        })

        with open('data/match_values_valid.json', 'w') as f:
            f.write(json.dumps(lst_output, indent=2, ensure_ascii=False))

    @staticmethod
    def __preprocess():
        config = global_config.gen_config(
            ['--mode', 'preproc', '--is-cached', 'false', '--data-root', 'data', '--is-cached', 'false', '--output',
             'data/preproc'])
        init_env(config)
        preprocess(config)

    def __inference(self):
        config = global_config.gen_config(
            ['--mode', 'infer', '--init-model-param', 'model/model.pdparams', '--valid-set', 'data/preproc/valid.pkl',
             '--data-root', 'data/preproc'])
        init_env(config)
        run_mode = config.general.mode
        _set_proc_name(config, run_mode)
        pred_query = inference(config, self.model)
        self.plainTextEdit_output.setPlainText(pred_query)

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
    """
    浙江500kV变电站的发电平均值分别是多少
    AI_SEARCH_28
    select t1.AVERAGE,t2.NAME from SUBSTATION_STATISTIC_POWER as t1 join SUBSTATION_BASIC as t2 on t1.ID = t2.ID where t2.REGION like '%浙江%' and t2.TYPE like '%变电站%' and t2.TOP_AC_VOLTAGE_TYPE = '500'
    """

    warnings.filterwarnings("ignore")

    app = QApplication(sys.argv)
    mainWin = mainWindow()

    mainWin.show()
    sys.exit(app.exec_())
