#!/usr/bin/env python
# coding: utf-8

import json
import os
import re
import xml.etree.ElementTree as ET

import openai
import feedparser
from bs4 import BeautifulSoup
from tqdm.auto import tqdm


if os.environ.get('OPENAI_AZURE_BASE') is not None:
    openai.api_base = os.environ.get('OPENAI_AZURE_BASE')
    openai.api_key = os.environ.get('OPENAI_AZURE_API_KEY')
    openai.api_type = "azure"
    openai.api_version = os.environ.get('OPENAI_AZURE_VERSION', "2022-12-01")
elif os.environ.get('OPENAI_API_KEY') is not None:
    openai.api_key = os.environ.get('OPENAI_API_KEY')
else:
    print('Please set OPENAI_API_KEY or OPENAI_AZURE_API_KEY and OPENAI_AZURE_BASE')
    exit(1)


prompt_temp='''<|im_start|>system
你是一个论文的翻译与摘要机器人，你会把用户输入的论文信息翻译成中文，然后把其中关于论文最重要的创新和贡献总结成一句话，
并把这些内容以JSON的格式输出，你不会写程序，你不会提供其他建议，你不会给出代码
你会用下面的格式输出信息：
translated_title: 这里是翻译过的论文标题
translated_abstract: 这里是翻译过的论文摘要
tldr: 这里是中文总结出的一句话要点
en_tdlr: 这里是英文总结出的一句话要点
<|im_end|>
<|im_start|>user
{article}
<|im_end|>
'''
output_dir = 'papers'


def try_load(answer):
    if '\\' in answer:
        answer = answer.replace('\\', '\\\\')
    return json.loads(answer)


def call_chat(prompt):
    final_ret = {}
    ret = None
    try:
        ret = openai.Completion.create(
            engine="gpt-35-turbo",
            prompt=prompt,
            temperature=0,
            max_tokens=2000,
            top_p=1.0,
            frequency_penalty=0,
            presence_penalty=0,
            stop=["<|im_end|>"])
        answer = ret['choices'][0]['text']
        # final_ret['raw_ret'] = answer
        final_ret['total_tokens'] = ret['usage']['total_tokens']

        # translated_title: 这里是翻译过的论文标题
        # translated_abstract: 这里是翻译过的论文摘要
        # tldr: 这里是中文总结出的一句话要点
        # en_tdlr: 这里是英文总结出的一句话要点

        for line in answer.split('\n'):
            if line.startswith('translated_title'):
                final_ret['translated_title'] = line.split(':', 1)[1].strip()
            if line.startswith('translated_abstract'):
                final_ret['translated_abstract'] = line.split(':', 1)[1].strip()
            if line.startswith('tldr'):
                final_ret['tldr'] = line.split(':', 1)[1].strip()
            if line.startswith('en_tdlr'):
                final_ret['en_tdlr'] = line.split(':', 1)[1].strip()

        return final_ret
    except KeyboardInterrupt:
        raise
    except:
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
        if 'tldr' in x:
            ind = len(summary) + 1
            tldr = x['tldr'].replace('\n', ' ')
            en_tldr = x.get('en_tldr', '').replace('\n', ' ')
            summary.append(f"| [^{ind}] | [{clean_title(x['title'])}]({x['link']}) | {tldr} |")
            ta = x['translated_abstract'].replace('\n', ' ')
            a = x['abstract'].replace('\n', ' ')
            details.append(f"""[^{ind}]: {x['translated_title']}

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
        if 'tldr' in x:
            item = ET.SubElement(channel, "item")
            item_title = ET.SubElement(item, "title")
            item_title.text = x['tldr']
            item_link = ET.SubElement(item, "link")
            item_link.text = x['link']
            item_desc = ET.SubElement(item, "description")

            ta = x['translated_abstract'].replace('\n', ' ')
            a = x['abstract'].replace('\n', ' ')
            item_desc.text = f"""<p>
{x['translated_title']}
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
    print('download feed', arxiv_channel)
    # Parse the arXiv.org RSS feed
    feed = feedparser.parse(f'https://export.arxiv.org/rss/{arxiv_channel}')
    rets = []
    for item in tqdm(feed.entries):
        arxiv_id = item.link.split('/')[-1]
        path = get_path(arxiv_id)
        if os.path.exists(path):
            ret = json.load(open(path, 'r'))
        else:
            soup = BeautifulSoup(item.description, 'html.parser')
            description_text = soup.get_text().strip()[:1000]
            description_text = description_text.replace('-\n', '').replace('\n', ' ')
            content = f'''Title: {item.title[:1000]}
Abstract: {description_text}'''
            prompt = prompt_temp.format(article=content)
            ret = call_chat(prompt)
            if 'tldr' not in ret:
                continue
            ret = {
                'title': item.title,
                'abstract': description_text,
                'link': item.link,
                **ret
            }
            path_dir = os.path.dirname(path)
            os.makedirs(path_dir, exist_ok=True)
            with open(path, 'w') as fp:
                json.dump(ret, fp, indent=4, ensure_ascii=False)
        rets.append(ret)

    rets = sorted(rets, key=lambda x: x['link'], reverse=True)
    markdown = make_markdown(rets)
    with open(f'{arxiv_channel}.md', 'w') as fp:
        fp.write(markdown)
    make_rss(rets, arxiv_channel)


if __name__ == '__main__':
    chat_arxiv('cs.AI')
    chat_arxiv('cs.CL')
    chat_arxiv('cs.LG')
    chat_arxiv('econ')
    chat_arxiv('q-fin')
