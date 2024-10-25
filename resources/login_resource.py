# auth/login.py

from flask import Blueprint, request, jsonify
from flask_jwt_extended import create_access_token
from config.login import User_name, User_password
from datetime import timedelta

login_bp = Blueprint('login', __name__)

@login_bp.route('/login', methods=['POST'])
def login():
    # 检查请求体是否为空
    if not request.json or 'username' not in request.json or 'password' not in request.json:
        return jsonify({"error": "Username and password are required."}), 400

    username = request.json.get('username')
    password = request.json.get('password')

    if username == User_name and password == User_password:
        # 生成访问令牌，设置过期时间
        access_token = create_access_token(identity=username, expires_delta=timedelta(minutes=30))
        return jsonify(access_token=access_token, expires_in=1800), 200
    else:
        return jsonify({"error": "Username or password error"}), 401
