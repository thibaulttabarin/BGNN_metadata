# File: main.py
import sys
import json
import pprint
import enum

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, Enum, Text
#from sqlalchemy.sql import select

from PySide6.QtUiTools import QUiLoader
from PySide6.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton
from PySide6.QtCore import QFile, QIODevice
from PySide6.QtGui import QPixmap

with open('./check_labels.json') as f:
    metadata = json.load(f)
#filename = None
#curr_metadata = None
#window = None

engine = create_engine('sqlite:///label_checking.sqlite')#, echo=True)
conn = engine.connect()
Session = sessionmaker(bind=engine)
session = Session()

Base = declarative_base()

class ErrTypes(enum.Enum):
    auto_wrong_ocr = 1
    auto_wrong_tag = 2
    auto_wrong_synonym = 3
    auto_wrong_other = 4
    both_wrong = 5
    auto_right = 6
    both_right = 7

class Record(Base):
    __tablename__ = 'results'

    id = Column(Integer, primary_key=True)
    filename = Column(String)
    sci_name = Column(String)
    err_type = Column('err_type', Enum(ErrTypes))
    description = Column(Text)

    def __repr__(self):
        return f'User {self.name}'

Base.metadata.create_all(engine)

def load_next():
    fname_gen = get_filename()
    global filename
    filename = fname_gen.__next__()
    print(filename)
    pixmap = QPixmap('images/check_labels_prediction_{}.png'
            .format(filename))
    window.picture_frame.setPixmap(pixmap)
    global curr_metadata
    curr_metadata = metadata[filename]
    window.label_text.setPlainText(curr_metadata['tag_text'])
    del curr_metadata['tag_text']
    window.metadata.setPlainText(pprint.pformat(curr_metadata))
    window.scientific_name.clear()
    window.further_descr.clear()
    print()
    done = session.query(Record).count()
    window.sys_status.setPlainText('Overall Total: {0}\nTotal to Check: {1}\nDone: {2}\nRemaining: {3}'
            .format(total, count, done, count - done))

def ocr_mistake():
    name = Record(filename=filename, sci_name=curr_metadata['metadata_name'].capitalize(),
            err_type=ErrTypes.auto_wrong_ocr, description=window.further_descr.toPlainText())
    session.add(name)
    session.commit()
    load_next()

def both_correct():
    name = Record(filename=filename, sci_name=curr_metadata['best_name'].capitalize(),
            err_type=ErrTypes.both_right, description=window.further_descr.toPlainText())
    session.add(name)
    session.commit()
    load_next()

def auto_correct():
    name = Record(filename=filename, sci_name=curr_metadata['best_name'].capitalize(),
            err_type=ErrTypes.auto_right, description=window.further_descr.toPlainText())
    session.add(name)
    session.commit()
    load_next()

def get_filename():
    for filename in metadata.keys():
        if 'errored' in metadata[filename].keys() or\
                ((not metadata[filename]['matched_metadata'])
                or metadata[filename]['lev_dist'] > 2):

            #s = select(Base)
            #s = select(Record).filter_by(filename=filename)
            result = session.query(Record).filter_by(filename=filename).all()
            if not result:
                yield filename

def main():
    app = QApplication(sys.argv)

    ui_file_name = "picture_viewer.ui"
    ui_file = QFile(ui_file_name)
    if not ui_file.open(QIODevice.ReadOnly):
        print(f"Cannot open {ui_file_name}: {ui_file.errorString()}")
        sys.exit(-1)
    loader = QUiLoader()
    global window
    window = loader.load(ui_file)
    ui_file.close()
    if not window:
        print(loader.errorString())
        sys.exit(-1)

    global count
    count = 0
    for key in metadata.keys():
        if 'errored' in metadata[key].keys():
            count += 1
        elif not metadata[key]['matched_metadata'] or\
                metadata[key]['lev_dist'] > 2:
            count += 1
    global total
    total = len(metadata.keys())

    load_next()

    #fname_gen = get_filename()
    #filename = fname_gen.__next__()
    #print(filename)
    #pixmap = QPixmap('images/check_labels_prediction_{}.png'
    #        .format(filename))
    #window.picture_frame.setPixmap(pixmap)
    #curr_metadata = metadata[filename]['tag_text']
    #window.label_text.setPlainText(curr_metadata)
    #del metadata[filename]['tag_text']
    #window.metadata.setPlainText(pprint.pformat(metadata[filename]))

    #with open('./check_labels_text.txt') as f:
        #labels = f.readlines()
    #i = 0
    #while i < len(labels):
        #if '725' in labels[i]:
            #temp = ''
            #i += 2
            #while 'Matches metadata' not in labels[i]:
                #temp = temp + labels[i]
                #i += 1
            #window.label_text.setPlainText(temp)
            #break
        #i += 1

    #window.metadata.setPlainText(pprint.pformat(metadata['INHS_FISH_725.jpg']))

    window.csv_wrong.clicked.connect(auto_correct)
    window.both_correct.clicked.connect(both_correct)
    window.ocr_mistake.clicked.connect(ocr_mistake)

    window.show()
    exit_code = app.exec()
    session.close()
    sys.exit(exit_code)

if __name__ == "__main__":
#if True:
    main()

