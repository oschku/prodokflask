from flask import Flask, render_template
import connexion
from flask_swagger_ui import get_swaggerui_blueprint
from project import create_app

#create the application instance
app = create_app()



if __name__ == "__main__":
    app.run(host = '0.0.0.0', port= 5000,  debug=True)