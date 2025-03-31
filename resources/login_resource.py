# auth/login.py

import logging
from flask import Blueprint, request, jsonify
from flask_jwt_extended import create_access_token
from config.login import USERS
from datetime import timedelta

# 设置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

login_bp = Blueprint('login', __name__)

@login_bp.route('/login', methods=['POST'])
def login():
    # 检查请求体是否为空
    if not request.json or 'username' not in request.json or 'password' not in request.json:
        return jsonify({"error": "Username and password are required."}), 400

    username = request.json.get('username')
    password = request.json.get('password')

    # 检查用户名和密码是否匹配
    if username in USERS and USERS[username] == password:
        # 生成访问令牌，设置过期时间
        access_token = create_access_token(identity=username, expires_delta=timedelta(minutes=30))
        
        # 记录成功登录信息
        logging.info(f"User '{username}' logged in successfully.")
        
        return jsonify(access_token=access_token, expires_in=1800), 200
    else:
        # 记录登录失败信息
        logging.warning(f"Failed login attempt for user '{username}'.")
        return jsonify({"error": "Username or password error"}), 401
