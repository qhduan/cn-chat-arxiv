#!/usr/bin/env python
# coding: utf-8

import json
import datetime
import os
import re
import traceback
import xml.etree.ElementTree as ET
from copy import deepcopy
from concurrent.futures import TimeoutError

import openai
import feedparser
from bs4 import BeautifulSoup
from tqdm.auto import tqdm
from pebble import concurrent, ProcessPool


if os.environ.get('OPENAI_AZURE_BASE') is not None:
    openai.api_base = os.environ.get('OPENAI_AZURE_BASE')
    openai.api_key = os.environ.get('OPENAI_AZURE_API_KEY')
    engine = os.environ.get('OPENAI_AZURE_ENGINE')
    openai.api_version = os.environ.get('OPENAI_AZURE_VERSION', "2022-12-01")
    openai.api_type = "azure"
elif os.environ.get('OPENAI_API_KEY') is not None:
    openai.api_key = os.environ.get('OPENAI_API_KEY')
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
output_dir = 'papers'


def try_load(answer):
    if '\\' in answer:
        answer = answer.replace('\\', '\\\\')
    return json.loads(answer)


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
                model="gpt-3.5-turbo",
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


def get_path(arxiv_id):
    path = os.path.join(output_dir, arxiv_id[:2], arxiv_id[2:4], arxiv_id + '.json')
    return path


def clean_title(x):
    x = re.sub(r'\s*\(arXiv.+', '', x)
    x = x.replace('\n', ' ')
    return x


def make_markdown(rets):
    summary = []
    details = []
    for x in rets:
        if 'tldr' in x and 'translated_title' in x and 'translated_abstract' in x:
            ind = len(summary) + 1
            tldr = x['tldr'].replace('\n', ' ')
            en_tldr = x.get('en_tldr', '').replace('\n', ' ')
            summary.append(f"| [^{ind}] | [{clean_title(x['title'])}]({x['link']}) | {tldr} |")
            tt = x.get('translated_title', '').replace('\n', ' ')
            ta = x.get('translated_abstract', '').replace('\n', ' ')
            a = x['abstract'].replace('\n', ' ')
            details.append(f"""[^{ind}]: {tt}

    {x['title']}

    [{x['link']}]({x['link']})

    {tldr}

    {en_tldr}

    {ta}

    {a}
    """)
    summary_text = '\n'.join(summary)
    details_text = '\n'.join(details)

    markdown = f'''# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
{summary_text}

# 详细

{details_text}

'''
    return markdown


def make_rss(rets, arxiv_channel='cs.AI'):
    # Create the root element
    rss = ET.Element("rss")
    rss.set("version", "2.0")

    # Create the channel element
    channel = ET.SubElement(rss, "channel")

    # Add required channel elements
    title = ET.SubElement(channel, "title")
    title.text = f"Chat Arxiv {arxiv_channel}"
    link = ET.SubElement(channel, "link")
    link.text = "https://github.com/qhduan/cn-chat-arxiv"
    description = ET.SubElement(channel, "description")
    description.text = f"This is arxiv RSS feed for {arxiv_channel}"

    # Add some items to the channel
    for x in rets:
        if 'tldr' in x and 'translated_title' in x and 'translated_abstract' in x:
            item = ET.SubElement(channel, "item")
            item_title = ET.SubElement(item, "title")
            item_title.text = x['tldr']
            item_link = ET.SubElement(item, "link")
            item_link.text = x['link']
            item_desc = ET.SubElement(item, "description")

            ta = x.get('translated_abstract', '').replace('\n', ' ')
            tt = x.get('translated_title', '').replace('\n', ' ')
            a = x['abstract'].replace('\n', ' ')
            item_desc.text = f"""<p>
{tt}
</p>
<p>
{x['title']}
</p>
<p>
{x['link']}
</p>
<p>
{x['tldr']}
</p>
<p>
{x.get('en_tldr', '')}
</p>
<p>
{ta}
</p>
<p>
{a}
</p>"""

    # Save the XML file
    tree = ET.ElementTree(rss)
    tree.write(f"{arxiv_channel}.xml")


def chat_arxiv(arxiv_channel='cs.AI'):
    """
    Download the arxiv feed and use ChatGPT to do summary
    """
    print('download feed', arxiv_channel)
    # Parse the arXiv.org RSS feed
    feed = feedparser.parse(f'https://export.arxiv.org/rss/{arxiv_channel}')
    print(f'we found {len(feed.entries)} items')
    to_call_chat = []
    good_rets = []
    for item in feed.entries:
        arxiv_id = item.link.split('/')[-1]
        path = get_path(arxiv_id)
        if os.path.exists(path):
            ret = json.load(open(path, 'r'))
            good_rets.append(ret)
        else:
            soup = BeautifulSoup(item.description, 'html.parser')
            description_text = soup.get_text().strip()[:1000]
            description_text = description_text.replace('-\n', '').replace('\n', ' ')
            context = f'''Title: {item.title[:1000]}
Abstract: {description_text}'''
            ret = {
                'title': item.title,
                'abstract': description_text,
                'link': item.link,
                'context': context,
                'path': path,
            }
            to_call_chat.append(ret)

    print(f'{len(to_call_chat)} paper need to chat')
    if len(to_call_chat) > 0:
        with ProcessPool(max_workers=min(len(to_call_chat), 32)) as pool:
            futures = []
            for ret in to_call_chat:
                future = pool.schedule(call_chat, [ret['context']], timeout=300)
                futures.append(future)
            for ret, f in tqdm(zip(to_call_chat, futures), total=len(futures)):
                try:
                    result = f.result()  # blocks until results are ready
                    if 'tldr' in result:
                        good_rets.append({
                            **ret,
                            **result
                        })
                except TimeoutError as error:
                    continue
                except Exception as error:
                    print(error)

    for ret in good_rets:
        if path in ret:
            path = ret['path']
        else:
            arxiv_id = ret['link'].split('/')[-1]
            path = get_path(arxiv_id)
        path_dir = os.path.dirname(path)
        os.makedirs(path_dir, exist_ok=True)
        with open(path, 'w') as fp:
            json.dump(ret, fp, indent=4, ensure_ascii=False)

    rets = sorted(good_rets, key=lambda x: x['link'], reverse=True)
    markdown = make_markdown(rets)
    with open(f'{arxiv_channel}.md', 'w') as fp:
        fp.write(markdown)
    make_rss(rets, arxiv_channel)
    with open('latest_updated.txt', 'w') as fp:
        fp.write(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))


if __name__ == '__main__':
    cs = '''AI,CL,LG,IR'''.split(',')
    for c in cs:
        chat_arxiv(f'cs.{c}')
    others = '''econ,q-fin,stat.ML'''.split(',')
    for c in others:
        chat_arxiv(c)
