#!/usr/bin/env python
# coding: utf-8

import json
import datetime
import os
import re
import sys
import time
import traceback
import xml.etree.ElementTree as ET
from copy import deepcopy
from concurrent.futures import TimeoutError

import feedparser
import requests
from bs4 import BeautifulSoup
from tqdm.auto import tqdm
from pebble import concurrent, ProcessPool
from llm import call_chat

output_dir = 'papers'

# arxiv 的 RSS 偶尔会返回空的 channel（例如停止 announcement、CDN 缓存尚未构建、
# 或者请求被限流）。这时绝对不能拿空结果去覆盖已有的 md/xml，否则会把线上内容清空。
FEED_URLS = (
    'https://rss.arxiv.org/rss/{channel}',
    'https://export.arxiv.org/rss/{channel}',
)
FEED_ATTEMPTS = int(os.environ.get('FEED_ATTEMPTS', '3'))
FEED_RETRY_WAIT = int(os.environ.get('FEED_RETRY_WAIT', '15'))

# 渲染 md/xml 需要的字段。历史上 LLM 偶尔少输出一行，脚本仍然把结果写盘，
# 于是留下了一批永远渲染不出来的“残件”。这些记录由 backfill.py 补全。
REQUIRED_FIELDS = ('translated_title', 'translated_abstract', 'tldr')


def missing_fields(record):
    return [k for k in REQUIRED_FIELDS if not record.get(k)]


def is_complete(record):
    return not missing_fields(record)


def fetch_feed(arxiv_channel):
    """下载并解析 RSS，返回条目列表。

    空 feed 或解析失败时换一个地址重试，仍未拿到内容则返回 None，表示
    “上游没数据”。调用方必须把它和“上游有数据但都命中缓存”区分开。
    """
    for attempt in range(1, FEED_ATTEMPTS + 1):
        url = FEED_URLS[(attempt - 1) % len(FEED_URLS)].format(channel=arxiv_channel)
        try:
            # 自己发请求：feedparser 直连不走 requests，超时不可控，
            # 上游卡住时会让 job 一直挂到 GitHub 的 6 小时后超时。
            resp = requests.get(url, timeout=60, headers={'Cache-Control': 'no-cache'})
            resp.raise_for_status()
            feed = feedparser.parse(resp.content)
            if feed.entries:
                return feed.entries
            print(f'{arxiv_channel}: 第 {attempt} 次请求得到 0 条 '
                  f'({url}, status={resp.status_code}, '
                  f'size={len(resp.content)}, version={feed.get("version")!r}, '
                  f'feed_updated={feed.feed.get("updated")!r}, '
                  f'skipdays={feed.feed.get("skipdays")!r})')
        except Exception as error:
            print(f'{arxiv_channel}: 第 {attempt} 次请求失败 ({url}): {error!r}')
        if attempt < FEED_ATTEMPTS:
            time.sleep(FEED_RETRY_WAIT)
    return None


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


def write_if_changed(path, content):
    """内容没变就不动文件，避免产生无意义的 diff 和提交。"""
    try:
        with open(path, 'r', encoding='utf-8') as fp:
            if fp.read() == content:
                return False
    except FileNotFoundError:
        pass
    path_dir = os.path.dirname(path)
    if path_dir:
        os.makedirs(path_dir, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as fp:
        fp.write(content)
    return True


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

    # 保持和原来 tree.write() 一致的 us-ascii 转义格式，避免无意义的 diff
    return ET.tostring(rss, encoding='us-ascii').decode('ascii')


def chat_arxiv(arxiv_channel='cs.AI'):
    """
    Download the arxiv feed and use ChatGPT to do summary

    返回 'changed'（有内容写入）、'unchanged'（上游有数据但没有变化）
    或 'empty'（上游没数据，本次完全跳过，不会覆盖任何已有文件）。
    """
    print('download feed', arxiv_channel)
    entries = fetch_feed(arxiv_channel)
    if entries is None:
        # 关键保护：上游空 feed 时直接跳过，绝不能用空结果覆盖已有的 md/xml，
        # 否则 RSS 订阅者会收到空 feed，仓库历史也会被清空。
        print(f'{arxiv_channel}: 连续 {FEED_ATTEMPTS} 次拿到空 feed，跳过本次更新')
        return 'empty'
    print(f'we found {len(entries)} items')

    to_call_chat = []
    good_rets = []
    for item in entries:
        arxiv_id = item.link.split('/')[-1]
        path = get_path(arxiv_id)
        if os.path.exists(path):
            ret = json.load(open(path, 'r'))
            if is_complete(ret):
                ret.pop('pending', None)
                good_rets.append(ret)
                continue
            # 历史残件：元数据都在，只缺 LLM 字段，排队补全
            print(f'{arxiv_id}: 缓存缺 {missing_fields(ret)}，重新摘要')
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

    # 先把元数据落盘，再调用 LLM：即使这次摘要全部失败，论文信息也已经保存，
    # 之后可以只靠 backfill.py 补字段，不用重新抓 feed。
    for ret in to_call_chat:
        if is_complete(ret):
            continue
        pending = {**ret, 'pending': missing_fields(ret)}
        write_if_changed(ret['path'], json.dumps(pending, indent=4, ensure_ascii=False))

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
                    merged = {k: v for k, v in {**ret, **result}.items() if k != 'pending'}
                    if is_complete(merged):
                        good_rets.append(merged)
                    else:
                        # 缺字段的结果不写进 md/xml（渲染不出来），但已拿到的字段
                        # 要存下来，backfill 之后只补剩下的部分。
                        print(f'{ret.get("link")}: 摘要缺 {missing_fields(merged)}，稍后补全')
                        merged['pending'] = missing_fields(merged)
                        write_if_changed(ret['path'], json.dumps(merged, indent=4, ensure_ascii=False))
                except TimeoutError as error:
                    continue
                except Exception as error:
                    print(error)

    if not good_rets:
        # 有条目但一篇都没摘要成功（例如 API 全部失败）：同样保留旧文件。
        print(f'{arxiv_channel}: 本次没有任何可用摘要，跳过写入')
        return 'empty'

    changed = False
    for ret in good_rets:
        arxiv_id = ret['link'].split('/')[-1]
        path = ret.get('path') or get_path(arxiv_id)
        changed |= write_if_changed(path, json.dumps(ret, indent=4, ensure_ascii=False))

    rets = sorted(good_rets, key=lambda x: x['link'], reverse=True)
    changed |= write_if_changed(f'{arxiv_channel}.md', make_markdown(rets))
    changed |= write_if_changed(f'{arxiv_channel}.xml', make_rss(rets, arxiv_channel))
    return 'changed' if changed else 'unchanged'


if __name__ == '__main__':
    channels = '''cs.AI,cs.CL,cs.LG,cs.IR,cs.SE,econ,q-fin,stat.ML'''.split(',')
    changed = []
    unchanged = []
    empty = []
    errors = []
    for c in channels:
        try:
            status = chat_arxiv(c)
        except Exception:
            # 单个频道异常不影响其它频道，最后统一让 CI 失败以便发现问题
            traceback.print_exc()
            print(f'{c}: 运行异常')
            errors.append(c)
            continue
        {'changed': changed, 'unchanged': unchanged, 'empty': empty}[status].append(c)

    print(f'完成：{len(changed)} 个频道有变化（{", ".join(changed) or "无"}），'
          f'{len(unchanged)} 个频道无变化（{", ".join(unchanged) or "无"}），'
          f'{len(empty)} 个频道上游为空（{", ".join(empty) or "无"}），'
          f'{len(errors)} 个频道异常（{", ".join(errors) or "无"}）')

    if changed:
        # 只有内容真的变了才刷新时间戳。上游停更期间它会停在最后一次更新的时刻，
        # 一眼就能看出是 arXiv 没更新而不是脚本挂了，也避免每 6 小时提交一个空改动。
        with open('latest_updated.txt', 'w') as fp:
            fp.write(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    elif len(empty) == len(channels):
        # 所有频道都拿不到上游数据：arXiv 停更或 CDN 异常。打印 annotation 方便在
        # Actions 页面上看到，但不让 CI 变红，避免上游停更期间一直报警。
        print('::warning::所有频道均未拿到 arXiv 数据（上游为空），本次未更新任何文件')

    if errors:
        sys.exit(1)
