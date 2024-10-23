# auth/login.py

from flask import Blueprint, request, jsonify
from flask_jwt_extended import create_access_token
from config.login import User_name, User_password

login_bp = Blueprint('login', __name__)

@login_bp.route('/login', methods=['POST'])
def login():
    username = request.json.get('username')
    password = request.json.get('password')

    if username == User_name and password == User_password:
        access_token = create_access_token(identity=username)
        return jsonify(access_token=access_token), 200
    else:
        return jsonify({"error": "Username or password error"}), 401