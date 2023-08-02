import os
import openai
import time
import json
import pickle
import threading
# openai.api_key = ""

class MyThread(threading.Thread):
    def __init__(self, func, args=()):
        super(MyThread, self).__init__()
        self.func = func
        self.args = args
    def run(self):
        time.sleep(1)
        self.result = self.func(*self.args)
    def get_result(self):
        threading.Thread.join(self)  # 等待线程执行完毕
        try:
            return self.result
        except Exception:
            return None

keys = ["", ""]

key = ""
openai.proxy = ""

def get_gpt_ans(model, prompt, key):
    # print(key)
    try:
        openai.api_key = key
        completion = openai.Completion.create(
                engine=model,
                temperature=0.7,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
                prompt=prompt,
                max_tokens = 220
                )
        ans = completion.choices[0].text.strip()
        time.sleep(22)
        # print(completion)
    except:
        print(key)
        time.sleep(22)
        return get_gpt_ans(model, prompt, key)
    return ans

def get_prompt(x):
    # prompt = "Please fill in the given template according to the given sentence, each [N] can fill in one or more words. Please read the sentence information and reduce the number of filled words as much as possible.\n"
    prompt = "Sentence: for those employees whose first name does not containing the letter m , display their total salary by binning the hire date into month interval for visualizing a bar chart , and display in descending by the y-axis .\n"
    prompt += "Template: Bar chart showing [N] in descending order of y-axis and binning by month\n"
    prompt += "Result: Bar chart showing salaries for employees whose first name does not contain \"m\" by hire date in descending order of y-axis and binning by month.\n"
    prompt += "Sentence: " + x['Sentence'] + '\n'
    prompt += "Template: " + x['Template'] + '\n'
    prompt += "Result: "
    return prompt
nl_temp = []    
with open("./nl.jsonl", "r", encoding="utf-8") as f:
    for row in f.readlines():
        nl_temp.append(json.loads(row))

# with open("./1.txt", "r", encoding="utf-8") as f:
#     convert_data = f.readlines()

# for i in convert_data:
#     print(i.strip('\n'))

index = 0
for i,x in enumerate(nl_temp[index:]):
    prompt = get_prompt(x)
    # print(prompt)
    # model = 'text-davinci-003'
    model = 'text-davinci-003'
    
    ans = get_gpt_ans(model, prompt, key)
    # for key in keys:
    #     t = MyThread(get_gpt_ans, (model, prompt, key,))
    #     t.start()
    #     ans = t.get_result()

    print(ans)

    with open(f"./{model}_data.jsonl", "a", encoding="utf-8") as f:
        json.dump({
            'id':x['id'],
            'question':ans
        }, f)
        f.write('\n')
