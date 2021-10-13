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

IMAGE_DIR = '/usr/local/bgnn/joel_pred_mask_images'
ERR_IMAGE_DIR = '/usr/local/bgnn/inhs_validation'

with open('./check_labels.json') as f:
    metadata = json.load(f)
#filename = None
#curr_metadata = None
#window = None

LEV_DIST_CUTOFF = 3

#engine = create_engine('sqlite:///label_checking.sqlite')#, echo=True)
engine = create_engine('sqlite:////usr/local/bgnn/label_checking.sqlite')#, echo=True)
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
    both_right_ocr_failed = 8
    both_right_tag_issue = 9

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
    global curr_metadata
    curr_metadata = metadata[filename]
    if 'errored' not in curr_metadata.keys():
        pixmap = QPixmap('{}/check_labels_prediction_{}.png'
                .format(IMAGE_DIR, filename))
        window.label_text.setPlainText(curr_metadata['tag_text'])
        del curr_metadata['tag_text']
    else:
        pixmap = QPixmap('{}/{}'
                .format(ERR_IMAGE_DIR, filename))
        window.label_text.clear()
        window.metadata.clear()
    window.metadata.setPlainText(pprint.pformat(curr_metadata))
    window.picture_frame.setPixmap(pixmap)
    window.scientific_name.clear()
    window.further_descr.clear()
    print()
    done = session.query(Record).count()
    window.sys_status.setPlainText(('Overall Total: {}\nTotal to Check: {}' +
                                    '\n\tErrored: {}\n\tDidn\'t Match: {}' +
                                    '\n\tLev Dist > {}: {}\nDone: {}' +
                                    '\nRemaining: {}')
            .format(total, count, errored, didnt_match, LEV_DIST_CUTOFF,
                    lev_dist_above, done, count - done))

def both_corr_tag_issue():
    name = Record(filename=filename, sci_name=curr_metadata['metadata_name'].capitalize(),
            err_type=ErrTypes.both_right_tag_issue, description=window.further_descr.toPlainText())
    session.add(name)
    session.commit()
    load_next()

def both_corr_ocr_failed():
    name = Record(filename=filename, sci_name=curr_metadata['metadata_name'].capitalize(),
            err_type=ErrTypes.both_right_ocr_failed, description=window.further_descr.toPlainText())
    session.add(name)
    session.commit()
    load_next()

def wrong_other():
    try:
        name = curr_metadata['metadata_name']
    except KeyError:
        name = "Errored out, must run program to debug"
    name = Record(filename=filename, sci_name=name,
            err_type=ErrTypes.auto_wrong_other, description=window.further_descr.toPlainText())
    session.add(name)
    session.commit()
    load_next()

def ocr_mistake():
    name = Record(filename=filename, sci_name=curr_metadata['metadata_name'].capitalize(),
            err_type=ErrTypes.auto_wrong_ocr, description=window.further_descr.toPlainText())
    session.add(name)
    session.commit()
    load_next()

def both_correct():
    name = Record(filename=filename, sci_name=curr_metadata['metadata_name'].capitalize(),
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
                or metadata[filename]['lev_dist'] > LEV_DIST_CUTOFF):

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
    global errored
    global didnt_match
    global lev_dist_above
    count, errored, didnt_match, lev_dist_above = 0, 0, 0, 0
    for key in metadata.keys():
        if 'errored' in metadata[key].keys():
            errored += 1
            count += 1
        elif not metadata[key]['matched_metadata']:
            didnt_match += 1
            count += 1
        elif metadata[key]['lev_dist'] > LEV_DIST_CUTOFF:
            lev_dist_above += 1
            count += 1
    global total
    total = len(metadata.keys())

    load_next()

    window.csv_wrong.clicked.connect(auto_correct)
    window.both_correct.clicked.connect(both_correct)
    window.ocr_mistake.clicked.connect(ocr_mistake)
    window.wrong_other.clicked.connect(wrong_other)
    window.both_correct_ocr_failed.clicked.connect(both_corr_ocr_failed)
    window.both_correct_tag_issue.clicked.connect(both_corr_tag_issue)

    window.show()
    exit_code = app.exec()
    session.close()
    sys.exit(exit_code)

if __name__ == "__main__":
#if True:
    main()

