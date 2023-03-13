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


openai.api_type = "azure"
openai.api_version = "2022-12-01"
openai.api_base = os.environ.get('OPENAI_AZURE_BASE')
openai.api_key = os.environ.get('OPENAI_AZURE_API_KEY')


prompt_temp='''<|im_start|>system
你是一个论文的翻译与摘要机器人，你会把用户输入的论文信息翻译成中文，然后把其中关于论文最重要的创新和贡献总结成一句话
你会用下面的JSON格式输出信息：
{{
    "translated_title": "这里是翻译过的论文标题",
    "translated_abstract": "这里是翻译过的论文摘要",
    "tldr": "这里是中文总结出的一句话要点"
}}
你会输出JSON作为答案，并且输出的JSON应该是符合标准的JSON文本
除了答案的JSON以外，你不会输出任何东西
输出用"{{"作为开头，用"}}"作为结尾，中间是一个合法的JSON格式
<|im_end|>
<|im_start|>user
{article}
<|im_end|>
'''
output_dir = 'papers'


def call_chat(prompt):
    final_ret = {}
    try:
        ret = response = openai.Completion.create(
            engine="gpt-35-turbo",
            prompt=prompt,
            temperature=1,
            max_tokens=2000,
            top_p=0.95,
            frequency_penalty=0,
            presence_penalty=0,
            stop=["<|im_end|>"])
        answer = ret['choices'][0]['text']
        final_ret['raw_ret'] = answer
        final_ret['total_tokens'] = ret['usage']['total_tokens']
        start = answer.find('{')
        end = answer.rfind('}')
        ret_obj = json.loads(answer[start:end + 1])
        final_ret['ret'] = ret_obj
    except KeyboardInterrupt:
        raise
    except:
        pass
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
    for x in rets[:50]:
        if 'ret' in x and 'tldr' in x['ret']:
            ind = len(summary) + 1
            tldr = x['ret']['tldr'].replace('\n', ' ')
            summary.append(f"| [^{ind}] | [{clean_title(x['title'])}]({x['link']}) | {tldr} |")
            ta = x['ret']['translated_abstract'].replace('\n', ' ')
            a = x['abstract'].replace('\n', ' ')
            details.append(f"""[^{ind}]: {x['ret']['translated_title']}

    {x['title']}

    [{x['link']}]({x['link']})

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


def make_rss(rets, arxiv_class):
    # Create the root element
    rss = ET.Element("rss")
    rss.set("version", "2.0")

    # Create the channel element
    channel = ET.SubElement(rss, "channel")

    # Add required channel elements
    title = ET.SubElement(channel, "title")
    title.text = f"Chinese Chat Arxiv {arxiv_class}"
    link = ET.SubElement(channel, "link")
    link.text = "https://github.com/qhduan/cn-chat-arxiv"
    description = ET.SubElement(channel, "description")
    description.text = f"This is arxiv RSS feed for cs.{arxiv_class}"

    # Add some items to the channel
    for x in rets[:50]:
        if 'ret' in x and 'tldr' in x['ret']:
            item = ET.SubElement(channel, "item")
            item_title = ET.SubElement(item, "title")
            item_title.text = x['ret']['tldr']
            item_link = ET.SubElement(item, "link")
            item_link.text = x['link']
            item_desc = ET.SubElement(item, "description")

            ta = x['ret']['translated_abstract'].replace('\n', ' ')
            a = x['abstract'].replace('\n', ' ')
            item_desc.text = f"""<p>
{x['ret']['translated_title']}
</p>
<p>
{x['title']}
</p>
<p>
[{x['link']}]({x['link']})
</p>
<p>
{ta}
</p>
<p>
{a}
</p>"""

    # Save the XML file
    tree = ET.ElementTree(rss)
    tree.write(f"cs.{arxiv_class}.xml")





def chat_arxiv(arxiv_class = 'AI'):
    # Parse the arXiv.org RSS feed
    feed = feedparser.parse(f'https://export.arxiv.org/rss/cs.{arxiv_class}')
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
    with open(f'cs.{arxiv_class}.md', 'w') as fp:
        fp.write(markdown)
    make_rss(rets, arxiv_class)


if __name__ == '__main__':
    chat_arxiv('AI')
    chat_arxiv('CL')
    chat_arxiv('LG')
