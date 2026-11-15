#!/usr/bin/env python
# coding: utf-8
"""
补全 papers/ 里字段残缺的记录。

历史上 LLM 偶尔少输出一行，脚本仍然把结果写盘，于是留下了一批缺
translated_title / translated_abstract / tldr 的 JSON。这些记录元数据
（title、abstract、link）齐全，但渲染不出 md/xml，等于没有收录。

用法：
    python backfill.py --dry-run   # 只统计缺什么，不调用 LLM
    python backfill.py             # 扫描并补全，单次最多 BACKFILL_LIMIT 条

设计成可增量重复运行：每次只处理一小批，跑完提交，下次从剩下的继续，
避免一次把 LLM 额度打满。跑完不需要手动重建 md/xml —— 这些论文下次
出现在 feed 里时会自动渲染。
"""

import argparse
import json
import os
import sys
from collections import Counter
from concurrent.futures import TimeoutError

from pebble import ProcessPool
from tqdm.auto import tqdm

from cn_chat_arxiv import (
    REQUIRED_FIELDS,
    is_complete,
    missing_fields,
    output_dir,
    write_if_changed,
)

# 单次运行最多补全多少条；可通过环境变量覆盖
BACKFILL_LIMIT = int(os.environ.get('BACKFILL_LIMIT', '200'))
BACKFILL_WORKERS = int(os.environ.get('BACKFILL_WORKERS', '16'))
BACKFILL_TIMEOUT = int(os.environ.get('BACKFILL_TIMEOUT', '300'))


def iter_records():
    """遍历所有缓存 JSON，产出 (path, record)。"""
    for root, _, files in os.walk(output_dir):
        for name in sorted(files):
            if not name.endswith('.json'):
                continue
            path = os.path.join(root, name)
            try:
                with open(path, 'r') as fp:
                    record = json.load(fp)
            except Exception as error:
                print(f'{path}: 读取失败 {error!r}')
                continue
            if not isinstance(record, dict):
                continue
            yield path, record


def collect_incomplete():
    return [(path, record) for path, record in iter_records() if not is_complete(record)]


def build_context(record):
    """用已有元数据重建调用 LLM 需要的 context。"""
    if record.get('context'):
        return record['context']
    return f"""Title: {record.get('title', '')[:1000]}
Abstract: {record.get('abstract', '')[:1000]}"""


def merge_result(record, result):
    """把 LLM 输出合进原记录；已有的非空字段保持不变。

    只补缺失项，是为了让补全结果和当年第一次生成的结果尽量一致，
    也避免把已经对的内容重写一遍。
    """
    merged = {k: v for k, v in record.items() if k != 'pending'}
    for key in REQUIRED_FIELDS + ('en_tdlr', 'total_tokens'):
        if merged.get(key):
            continue
        value = result.get(key)
        if value:
            merged[key] = value
    return merged


def backfill_one(record):
    # 必须在 worker 里导入：llm 在 import 时就检查 API key 并可能 exit，
    # 而 --dry-run 不需要 key。
    from llm import call_chat
    return call_chat(build_context(record))


def save(path, record):
    if is_complete(record):
        record.pop('pending', None)
    else:
        record['pending'] = missing_fields(record)
    return write_if_changed(path, json.dumps(record, indent=4, ensure_ascii=False))


def run_backfill(limit, dry_run):
    incomplete = collect_incomplete()
    print(f'发现 {len(incomplete)} 条残缺记录')

    if dry_run:
        counter = Counter()
        for _, record in incomplete:
            for key in missing_fields(record):
                counter[key] += 1
        for key, count in counter.most_common():
            print(f'  缺 {key:20} {count} 条')
        return 0

    if not incomplete:
        return 0

    batch = incomplete[:limit]
    print(f'本次处理前 {len(batch)} 条（上限 {limit}）')

    done = failed = 0
    with ProcessPool(max_workers=min(len(batch), BACKFILL_WORKERS)) as pool:
        futures = [pool.schedule(backfill_one, [record], timeout=BACKFILL_TIMEOUT)
                   for _, record in batch]
        for (path, record), future in tqdm(zip(batch, futures), total=len(futures)):
            try:
                result = future.result()
            except TimeoutError:
                print(f'{path}: 超时')
                failed += 1
                continue
            except Exception as error:
                print(f'{path}: {error!r}')
                failed += 1
                continue

            merged = merge_result(record, result)
            save(path, merged)
            if is_complete(merged):
                done += 1
            else:
                print(f'{path}: 仍缺 {missing_fields(merged)}，保留待下轮重试')
                failed += 1

    remaining = len(incomplete) - done
    print(f'补全 {done} 条，失败 {failed} 条，剩余 {max(remaining, 0)} 条')
    return 0 if done or not batch else 1


def main():
    parser = argparse.ArgumentParser(description='补全 papers/ 中残缺的 LLM 字段')
    parser.add_argument('--dry-run', action='store_true', help='只统计缺失字段，不调用 LLM')
    parser.add_argument('--limit', type=int, default=BACKFILL_LIMIT, help='本次最多补全多少条')
    args = parser.parse_args()
    return run_backfill(args.limit, args.dry_run)


if __name__ == '__main__':
    sys.exit(main())