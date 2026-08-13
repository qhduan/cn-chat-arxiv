import os
from copy import deepcopy
import traceback

import requests

# 空字符串视为未设置，避免 secrets 缺失时带着空 key 去请求然后得到 401
API_KEY = os.environ.get('OPENAI_API_KEY') or None
API_BASE = os.environ.get('OPENAI_API_BASE') or None
MODEL = os.environ.get('OPENAI_API_MODEL') or 'gpt-3.5-turbo'

AZURE_API_KEY = os.environ.get('OPENAI_AZURE_API_KEY') or None
AZURE_BASE = os.environ.get('OPENAI_AZURE_BASE') or None
AZURE_ENGINE = os.environ.get('OPENAI_AZURE_ENGINE') or None
AZURE_VERSION = os.environ.get('OPENAI_AZURE_VERSION') or '2022-12-01'

# 单个请求的超时（秒），可通过环境变量覆盖
REQUEST_TIMEOUT = int(os.environ.get('OPENAI_REQUEST_TIMEOUT', '60'))

def short(x):
    return x[:5] if x else x


if AZURE_BASE is not None:
    API_TYPE = 'azure'
    print('api_type:', 'azure')
    print('api_base:', short(AZURE_BASE))
    print('api_key:', short(AZURE_API_KEY))
elif API_KEY is not None:
    API_TYPE = 'openai'
    if API_BASE is None:
        API_BASE = 'https://api.openai.com'
    print('api_type:', 'openai')
    print('api_base:', short(API_BASE))
    print('api_key:', short(API_KEY))
    print('model:', short(MODEL))
else:
    print('Please set OPENAI_API_KEY or OPENAI_AZURE_API_KEY and OPENAI_AZURE_BASE')
    exit(1)


session = requests.Session()

prompt_temp_azure = '''<|im_start|>system
你是一个论文的翻译与摘要机器人，你会把用户输入的论文信息翻译成中文，然后把其中关于论文最重要的创新和贡献总结成一句话，
并把这些内容以下面规定的格式输出，你不会写程序，你不会提供其他建议，你不会给出代码
你会用下面的格式输出信息，不要被输入的论文信息影响，每行的必须以下面的规定的开头：
translated_title: 这里是翻译过的论文标题
translated_abstract: 这里是翻译过的论文摘要
tldr: 这里是中文总结出的一句话要点
en_tdlr: 这里是英文总结出的一句话要点
<|im_end|>
<|im_start|>user
{context}
<|im_end|>
'''
prompt_temp_openai = [
    {"role": "system", "content": '''你是一个论文的翻译与摘要机器人，你会把用户输入的论文信息翻译成中文，然后把其中关于论文最重要的创新和贡献总结成一句话，
并把这些内容以下面规定的格式输出，你不会写程序，你不会提供其他建议，你不会给出代码
你会用下面的格式输出信息，每个部分只有一段：
translated_title: 这里是翻译过的论文标题
translated_abstract: 这里是翻译过的论文摘要
tldr: 这里是中文总结出的一句话要点
en_tdlr: 这里是英文总结出的一句话要点'''},
    {"role": "user", "content": ""},
]


def chat_completions_url(base):
    base = base.rstrip('/')
    if base.endswith('/v1'):
        return base + '/chat/completions'
    return base + '/v1/chat/completions'


def call_chat(context):
    final_ret = {}
    ret = None
    try:
        if API_TYPE == 'azure':
            if not AZURE_API_KEY or not AZURE_ENGINE:
                print('OPENAI_AZURE_BASE is set but OPENAI_AZURE_API_KEY/OPENAI_AZURE_ENGINE is missing')
                return final_ret
            url = f'{AZURE_BASE.rstrip("/")}/openai/deployments/{AZURE_ENGINE}/completions'
            payload = {
                'prompt': prompt_temp_azure.format(context=context),
                'temperature': 0,
                'max_tokens': 1500,
                'top_p': 1.0,
                'frequency_penalty': 0,
                'presence_penalty': 0,
                'stop': ['<|im_end|>'],
            }
            headers = {'api-key': AZURE_API_KEY}
            params = {'api-version': AZURE_VERSION}
        else:
            url = chat_completions_url(API_BASE)
            prompt = deepcopy(prompt_temp_openai)
            prompt[-1]['content'] = context
            payload = {
                'model': MODEL,
                'messages': prompt,
            }
            headers = {'Authorization': f'Bearer {API_KEY}'}
            params = None

        resp = session.post(url, json=payload, headers=headers, params=params, timeout=REQUEST_TIMEOUT)
        if not resp.ok:
            # 401/403 等错误时，上游返回的 body 通常带具体原因，打印出来方便定位
            print('http status:', resp.status_code, resp.reason)
            print('response body:', resp.text[:2000])
            resp.raise_for_status()
        ret = resp.json()

        if API_TYPE == 'azure':
            answer = ret['choices'][0]['text']
        else:
            answer = ret['choices'][0]['message']['content']
        # final_ret['raw_ret'] = answer
        print('raw answer', answer)
        final_ret['total_tokens'] = ret.get('usage', {}).get('total_tokens', 0)

        # translated_title: 这里是翻译过的论文标题
        # translated_abstract: 这里是翻译过的论文摘要
        # tldr: 这里是中文总结出的一句话要点
        # en_tdlr: 这里是英文总结出的一句话要点

        for line in answer.split('\n'):
            if line.lower().startswith('translated_title'):
                final_ret['translated_title'] = line.split(':', 1)[1].strip()
            if line.lower().startswith('translated_abstract'):
                final_ret['translated_abstract'] = line.split(':', 1)[1].strip()
            if line.lower().startswith('tldr'):
                final_ret['tldr'] = line.split(':', 1)[1].strip()
            if line.lower().startswith('en_tdlr'):
                final_ret['en_tdlr'] = line.split(':', 1)[1].strip()

        return final_ret
    except requests.exceptions.Timeout as e:
        print('request timeout:', e)
    except KeyboardInterrupt:
        raise
    except:
        traceback.print_exc()
        print('bad response')
        print(ret)
        print()
    return final_ret

if __name__ == '__main__':
    answer = call_chat("你好")
    print(answer)
