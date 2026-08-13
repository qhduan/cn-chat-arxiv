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

import feedparser
from bs4 import BeautifulSoup
from tqdm.auto import tqdm
from pebble import concurrent, ProcessPool
from llm import call_chat

output_dir = 'papers'

def try_load(answer):
    if '\\' in answer:
        answer = answer.replace('\\', '\\\\')
    return json.loads(answer)

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
    cs = '''AI,CL,LG,IR,SE'''.split(',')
    for c in cs:
        chat_arxiv(f'cs.{c}')
    others = '''econ,q-fin,stat.ML'''.split(',')
    for c in others:
        chat_arxiv(c)
