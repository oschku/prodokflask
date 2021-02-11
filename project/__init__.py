from flask_sqlalchemy import SQLAlchemy
import sys
import config
from .bin import valuation
import connexion
from flask import current_app
from pathlib import Path

db = SQLAlchemy()

def create_app():
    """Construct the core app object."""
    
    app = connexion.App(__name__, specification_dir='./')
    p = str(Path(__file__).parents[1])
    app.add_api(p + '/specfile.yml')

    application = app.app
    application.config.from_object(config.Config)


    #force app to read swagger.yml file to configure the endpoints
    

    SWAGGER_URL = '/api/ui'

    # Initialize Plugins
    db.init_app(application)
   
    

    with application.app_context():
        from . import routes
        from .bin import valuation

        # Register Blueprints
        application.register_blueprint(routes.main_bp)

        # Create Database Models
        db.create_all()

        return application