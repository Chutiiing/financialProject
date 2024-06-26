import os
import sys
import pandas as pd
import sqlalchemy

class DatabaseReader:

    def __init__(self):

        self.stockmultifactor_engine = self.initialize_stockmultifactor_engine()
        self.mikuang_engine = self.initialize_mikuang_engine()
        self.gogoalest_engine = self.initialize_gogoalest_engine()
        self.hthink_royalflush_engine = self.initialize_hthink_royalflush_engine()
        self.super_symmetry_engine = self.initialize_super_symmetry_engine()

    def __del__(self):
        self.stockmultifactor_engine.dispose()
        self.mikuang_engine.dispose()
        self.gogoalest_engine.dispose()
        self.hthink_royalflush_engine.dispose()
        self.super_symmetry_engine.dispose()

    @staticmethod
    def initialize_stockmultifactor_engine():
        return sqlalchemy.create_engine('mysql+pymysql://root:xs@2023@10.72.65.126:3306/socketmutifactor?charset=utf8')

    @staticmethod
    def initialize_mikuang_engine():
        return sqlalchemy.create_engine("mysql+pymysql://root:xs@2023@10.72.65.126:3306/mikuang_new?charset=utf8")

    @staticmethod
    def initialize_gogoalest_engine():
        return sqlalchemy.create_engine("mysql+pymysql://root:xs@2023@10.72.65.126:3306/gogoalest3?charset=utf8")

    @staticmethod
    def initialize_hthink_royalflush_engine():
        return sqlalchemy.create_engine("mysql+pymysql://root:xs@2023@10.72.65.126:3306/hthink_royalflush?charset=utf8")

    @staticmethod
    def initialize_super_symmetry_engine():
        return sqlalchemy.create_engine("mysql+pymysql://root:xs@2023@10.72.65.126:3306/super_symmetry?charset=utf8")

    def read_from_db(self, dbname=None, sql=None):

        try:
            if dbname == "stockmultifactor":
                connection = self.stockmultifactor_engine.connect()
            elif dbname == 'gogoalest3':
                connection = self.gogoalest_engine.connect()
            elif dbname == 'mikuang_new':
                connection = self.mikuang_engine.connect()
            elif dbname == 'hthink_royalflush':
                connection = self.hthink_royalflush_engine.connect()
            elif dbname == 'super_symmetry':
                connection = self.super_symmetry_engine.connect()

            else:
                print("Invalid Database.")
                sys.exit(0)

            data = pd.read_sql_query(sql, con=connection)

        except:
            if dbname == "stockmultifactor":
                self.stockmultifactor_engine = self.initialize_stockmultifactor_engine()
                connection = self.stockmultifactor_engine.connect()
            elif dbname == 'gogoalest3':
                self.gogoalest_engine = self.initialize_gogoalest_engine()
                connection = self.gogoalest_engine.connect()
            elif dbname == 'mikuang_new':
                self.mikuang_engine = self.initialize_mikuang_engine()
                connection = self.mikuang_engine.connect()
            elif dbname == 'hthink_royalflush':
                self.hthink_royalflush_engine = self.initialize_hthink_royalflush_engine()
                connection = self.hthink_royalflush_engine.connect()
            elif dbname == 'super_symmetry':
                self.super_symmetry_engine = self.initialize_super_symmetry_engine()
                connection = self.super_symmetry_engine.connect()
            else:
                print("Invalid Database.")
                sys.exit(0)

            data = pd.read_sql_query(sql, con=connection)

        finally:
            connection.close()

        return data
