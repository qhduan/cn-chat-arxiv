import os
from copy import deepcopy
import openai
import traceback

if os.environ.get('OPENAI_AZURE_BASE') is not None:
    openai.api_base = os.environ.get('OPENAI_AZURE_BASE')
    openai.api_key = os.environ.get('OPENAI_AZURE_API_KEY')
    engine = os.environ.get('OPENAI_AZURE_ENGINE')
    openai.api_version = os.environ.get('OPENAI_AZURE_VERSION', "2022-12-01")
    openai.api_type = "azure"
    print('openai.api_type:', 'azure')
    print('openai.api_base:', openai.api_base[:5])
    print('openai.api_key:', openai.api_key[:5])
elif os.environ.get('OPENAI_API_KEY') is not None:
    openai.api_key = os.environ.get('OPENAI_API_KEY')
    if os.environ.get('OPENAI_API_BASE') is not None:
        openai.api_base = os.environ.get('OPENAI_API_BASE')
    openai_model = "gpt-3.5-turbo"
    if os.environ.get('OPENAI_API_MODEL') is not None:
        openai_model = os.environ.get('OPENAI_API_MODEL')
    print('openai.api_type:', 'openai')
    print('openai.api_base:', openai.api_base[:5])
    print('openai.api_key:', openai.api_key[:5])
    print('openai_model:', openai_model[:5])
else:
    print('Please set OPENAI_API_KEY or OPENAI_AZURE_API_KEY and OPENAI_AZURE_BASE')
    exit(1)


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

def call_chat(context):
    final_ret = {}
    ret = None
    try:
        if openai.api_type == 'azure':
            prompt = prompt_temp_azure.format(context=context)
            ret = openai.Completion.create(
                engine=engine,
                prompt=prompt,
                temperature=0,
                max_tokens=1500,
                top_p=1.0,
                frequency_penalty=0,
                presence_penalty=0,
                stop=["<|im_end|>"])
            answer = ret['choices'][0]['text']
        else:
            prompt = deepcopy(prompt_temp_openai)
            prompt[-1]['content'] = context
            ret = openai.ChatCompletion.create(
                model=openai_model,
                messages=prompt
            )
            answer = ret['choices'][0]['message']['content']
        # final_ret['raw_ret'] = answer
        print('raw answer', answer)
        final_ret['total_tokens'] = ret['usage']['total_tokens']

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
