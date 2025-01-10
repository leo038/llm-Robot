import time

from flask import Flask, request, jsonify

from robot_function import SmartRobot

smart_robot = SmartRobot()

app = Flask(__name__)


@app.route('/', methods=['GET'])
def home():
    return "Hello, World!"


@app.route('/post', methods=['POST'])
def robot_service():
    data = request.get_json()  # 获取 JSON 格式请求体中的数据
    if not data:
        return jsonify({"error": "No data provided"}), 400

    print(f"请求数据：{data}")

    fuction_list = data.get("function")

    for func in fuction_list:
        func_smart_robot = "smart_robot." + func
        print(f"机器人执行指令： {func_smart_robot}")
        eval(func_smart_robot)
        time.sleep(1)


    return jsonify(data)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
