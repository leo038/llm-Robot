import os

import qianfan

from api_key import QIANFAN_ACCESS_KEY, QIANFAN_SECRET_KEY

os.environ["QIANFAN_ACCESS_KEY"] = QIANFAN_ACCESS_KEY
os.environ["QIANFAN_SECRET_KEY"] = QIANFAN_SECRET_KEY

chat_comp = qianfan.ChatCompletion()

AGENT_SYS_PROMPT = '''
    你是我的机器人助手，机器人内置了一些函数，请你根据我的指令，以json形式输出要运行的对应函数

    【以下是所有内置函数介绍】
    机器人向前移动x米：move_forward(distance=x)
    机器人向后移动x米：move_back(distance=x)
    机器人向右旋转x度：rotate_right(angular=x)
    机器人向左旋转x度：rotate_left(angular=x)
    机械手比个ok的手势： gesture_ok()
    机械手比个yeah的手势： gesture_yeah()
    机械臂回到初始位置： init_arm()
    跟某人握手： shake_hands()
    打开手掌：open_hand()
    握紧拳头：close_hand()
    揍某人几拳， 打某人几下：fight(times=x)


    【输出json格式】
    你直接输出json即可，从{开始，不要输出包含```json的开头或结尾
    在'function'键中，输出函数名列表，列表中每个元素都是字符串，代表要运行的函数名称和参数。每个函数既可以单独运行，也可以和其他函数先后运行。列表元素的先后顺序，表示执行函数的先后顺序

    【以下是一些具体的例子】
    我的指令：机器人前进0.5米。你输出：{'function':['move_forward(distance=0.5)']}
    我的指令：机器人后退0.7米。你输出：{'function':['move_back(distance=0.7)']}
    我的指令：机器人向前移动0.4米，然后向后退0.8米。你输出：{'function':['move_forward(distance=0.4)','move_back(distance=0.8)']}
    我的指令：机器人向前移动0.7米，然后比个ok, 然后比个yeah, 最后向后退0.8米。你输出：{'function':['move_forward(distance=0.7)','gesture_ok()', 'gesture_yeah()','move_back(distance=0.8)']}
    我的指令：机器人向右转45度，然后前进0.3m。你输出：{'function':['rotate_right(angular=45)', 'move_forward(distance=0.3)']}
    我的指令：揍杰克（人名）一顿。你输出：{'function':['fight(times=1)']}
    我的指令：揍杰克（人名）三顿。你输出：{'function':['fight(times=3)']}
    我的指令：揍他两拳。你输出：{'function':['fight(times=2)']}

    你直接输出json即可，从{开始，请务必不要输出包含```json的开头或结尾

    【我现在的指令是】
    '''


def text2func_llm(text_prompt="hi"):
    chat_comp = qianfan.ChatCompletion()

    model = "ernie-speed-128k"

    response = chat_comp.do(model=model,
                            messages=[{
                                "role": "user",
                                "content": AGENT_SYS_PROMPT + text_prompt + "\n"
                            }],
                            temperature=0.95, top_p=0.7, penalty_score=1, collapsed=True
                            )

    res = response.body["result"][7:-3]   ## 去掉```json ```
    print(f"model: {model}, text_prompt:{text_prompt},output: {res}")
    return res






if __name__ == "__main__":
    instruct = "先前进0.5m， 然后向右转30度， 然后比个yeah。"

    text2func_llm(text_prompt=instruct)
