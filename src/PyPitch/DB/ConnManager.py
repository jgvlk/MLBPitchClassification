import os
from urllib.parse import quote_plus

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


db_conn_str = r'DRIVER={/usr/local/Cellar/msodbcsql17/17.6.1.1/lib/libmsodbcsql.17.dylib};SERVER=10.0.1.3;DATABASE=MLBPitchClassification;UID=svc_MLBPitchClassification;PWD=datascience;'
db_conn_str = quote_plus(db_conn_str)

engine = create_engine("mssql+pyodbc:///?odbc_connect=%s" % db_conn_str)

Session = sessionmaker(bind=engine)

class SessionManager(object):
    def __init__(self):
        self.session = Session()
        self.engine = engine

