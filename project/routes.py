from flask import Blueprint, render_template, redirect, url_for


# Blueprint Configuration
main_bp = Blueprint(
    'main_bp', __name__,
    template_folder='templates',
    static_folder='static'
)


@main_bp.route('/', methods=['GET'])
def home():
    return render_template('home.html')