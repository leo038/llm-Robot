import requests

from llm_function import text2func_llm

from audio import record_auto, speech_recognition


url = "http://192.168.2.194:5000" + "/post"


def robot_execute(function={'function': ['move_forward(distance=0.5)', 'move_back(distance=0.5)', 'gesture_ok()', 'gesture_yeah()']}):
    res = requests.post(url=url, json=function)





if __name__ == "__main__":

    record_auto()

    text = speech_recognition()


    # text = "左转30度， 然后后退0.5m, 最后比个ok。"


    function = text2func_llm(text_prompt=text)

    if function == {}:
        function ="{'function':['gesture_yeah()']}"


    robot_execute(function=eval(function))