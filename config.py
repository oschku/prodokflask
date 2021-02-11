# -*- coding: utf-8 -*-

"""Flask app configuration."""
from os import environ, path
from dotenv import load_dotenv
import os as os
from datetime import timedelta

basedir = path.abspath(path.dirname(__file__))
load_dotenv(path.join(basedir, '.env'))


osname = os.name




class Config:
    """Set configuration from environment variables."""


    def get_env_variable(name):
        try:
            return os.environ[name]
        except KeyError:
            message = "Expected environment variable '{}' not set.".format(name)
            raise Exception(message)

    # POSTGRES 
    POSTGRES_URL = get_env_variable("POSTGRES_URL")
    POSTGRES_USER = get_env_variable("POSTGRES_USER")
    POSTGRES_PW = get_env_variable("POSTGRES_PW")
    POSTGRES_DB = get_env_variable("POSTGRES_DB")


    # Flask-SQLAlchemy
    DB_URL = 'postgresql+psycopg2://{user}:{pw}@{url}/{db}'.format(user=POSTGRES_USER,pw=POSTGRES_PW,url=POSTGRES_URL,db=POSTGRES_DB)
    SQLALCHEMY_DATABASE_URI = DB_URL
    SQLALCHEMY_ECHO = False
    SQLALCHEMY_TRACK_MODIFICATIONS = False



    # Static Assets
    STATIC_FOLDER = 'static'
    TEMPLATES_FOLDER = 'templates'


    # Work-directory
    # WORK_DIR = get_env_variable('WORK_DIR')
    if osname == 'nt':
        os.environ['WORK_DIR'] = 'local'
    else:
        os.environ['WORK_DIR'] = 'docker'