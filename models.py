#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2017-04-13 10:04:32
# @Author  : Your Name (you@example.org)
# @Link    : http://example.org
# @Version : $Id$
import os
import time
import copy
import types
import hashlib
import binascii
import threading
import pandas as pd
from inspect import signature, isfunction
from functools import wraps
from traceback import format_exc
from heapq import *
from operator import itemgetter
from contextlib import contextmanager
from threading import Semaphore, Thread, Event, Lock
try:
    from urllib.parse import unquote
except ImportError:
    from urllib import unquote


class Dict(dict):
    '''支持a.b和deepcopy操作的类字典'''

    def __new__(cls, *args, **kwargs):
        return super().__new__(cls, *args, **kwargs)

    def __getattr__(self, x):
        return self[x]

    def __setattr__(self, x, y):
        self[x] = y        

    def __copy__(self):
        return Dict(copy.copy(dict(self)))


    def __deepcopy__(self, memo):
        return Dict(copy.deepcopy(dict(self)))


class ProcessPool(object):
    """进程池"""

    def __init__(sefl, processes=10):
        pass

    def apply_async(self):
        pass


class SharedCounter:

    def __init__(self):
        self.lock = threading.Lock()
        self.value = 999999999

    def incr(self):
        while True:
            with self.lock:
                print('Before incr: {}'.format(self.value))
                self.value += 1
                time.sleep(1)
                print('After incr: {}'.format(self.value))

    def decr(self):
        while True:
            with self.lock:
                self.value -= 1

    def run(self):
        t = threading.Thread(target=self.decr)
        t.start()
        t = threading.Thread(target=self.incr)
        t.start()


class EventThread:
    """基于事件的线程"""

    def __init__(self):
        self.event = Event()
        self.count = 0

    def start_task(self):
        self.event.wait()
        self.count += 1
        print('count: {}\n'.format(self.count))

    def run(self):
        for i in range(10):
            t = Thread(target=self.start_task)
            t.start()

    def start(self):
        self.event.set()


class SemaphoreThread:
    """基于信号量的线程"""

    def __init__(self):
        self.sema = Semaphore()
        self.count = 0

    def start_task(self):
        self.sema.acquire()
        self.count += 1
        print('coutn: {}\n'.format(self.count))

    def run(self):
        for i in range(10):
            t = Thread(target=self.start_task)

    def next(self):
        self.sema.release()


class CSVRender(object):
    """CSV解析工具, 用于格式化检查, 依赖pandas"""

    def __init__(self, csv_file=None):
        if csv_file is not None:
            self._df = pd.read_csv(csv_file)

    def read_csv(self):
        pass

    def all(self, index=0):
        '''根据key生成嵌套对象'''
        rst = {}
        key = self._df.columns[index]
        for i, _ in enumerate(self._df[key]):
            s = self._df.ix[i]
            key_id = s[key]
            d = {}
            for k, v in s.items():
                if v == key_id:
                    continue
                if isinstance(v, str):
                    v = eval(v)
                d[k] = v

            rst[key_id] = Dict(d)
        return rst



    def get_csv_files(self):
        '''读取当前目录的csv文件'''
        self.paths = [i for i in os.listdir('.') if i.endswith('.csv')]

    def for_each_get(self, column='Drop', ignroe='drop.csv'):
        '''遍历文件, 整合数据'''
        for file_name in self.paths:
            if file_name == ignroe:
                continue
            df = pd.read_csv(os.path.abspath(file_name))
            if column not in df.columns:
                continue
            yield file_name, list(df[column])

    def update_orign_file(self):
        ''''''
        df = pd.read_csv('drop.csv')
        id_to_index = {i: v for i, v in enumerate(df.Id)}
        for k, v in self.for_each_get():
            if k in id_to_index:
                n = id_to_index[k]
                df.Items.ix[n] = v
            else:
                d = {
                    'Id': os.path.splitext(k)[0],
                    "Items": v,
                    "Type": 0,
                    "Count": []
                }
                df = df.append(d, ignore_index=True)
        df.to_csv('new_drop.csv', index=False)

    @staticmethod
    def check_column(csv_file, column, fn):
        if not isfunction(fn):
            return False
        pd.read_csv(csv_file)


if __name__ == '__main__':
    pass
