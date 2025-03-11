from flask import Flask
from models import BalloonDataCache

app = Flask(__name__)
balloon_cache = BalloonDataCache()

from . import routes

def create_app():
    return app 