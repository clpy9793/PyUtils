#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2017-04-06 09:13:04
# @Author  : Your Name (you@example.org)
# @Link    : http://example.org
# @Version : $Id$
import os
import time
import copy
import types
import pandas
import hashlib
import binascii
import threading
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







@contextmanager
def closing(thing):
    try:
        yield thing
    finally:
        thing.close()


@contextmanager
def time_sign():
    print('{}'.format(time.asctime()))
    yield
    print('\n{}'.format(time.asctime()))


def fun(d):
    return sorted(d.items(), key=lambda x: x)


def gen_shandw_sign(data, key=""):
    """sdw"""
    data = {str(k): v for k, v in data.items()}
    rst = sorted(data.items(), key=lambda x: x[0])
    rst = ["{}={}".format(i[0], i[1])
           for i in rst if (i[0] != 'memo' and i[1])]
    print(rst)
    stringA = "&".join(rst)
    stringSignTemp = "{}{}".format(stringA, key)
    signValue = hashlib.md5(stringSignTemp.encode()).hexdigest().lower()
    return signValue


def get_rank_count():
    """计算当前时间对应的场次"""
    t = time.localtime()
    return t.tm_hour * 2 + int(t.tm_min / 30) + 1


def get_seconds():
    """计算距离指定时间点剩余的时间"""
    t = time.localtime()
    minute = t.tm_min % 30
    return (30 - minute) * 60 - t.tm_sec


class NoMixedCaseMeta(type):
    """拒绝大小写混合的类成员"""

    def __new__(cls, clsname, bases, clsdict):
        for name in clsdict:
            if name.lower() != name:
                raise TypeError('bad attribute name: {}'.format(name))
        return super().__new__(cls, clsname, bases, clsdict)


class Singleton(type):
    """单例"""

    def __init__(self, *args, **kwargs):
        self.__instance = None
        super().__init__(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        if self.__instance is None:
            self.__instance = super().__call__(*args, **kwargs)
        return self.__instance


class Profiled:
    """添加函数调用次数的统计"""

    def __init__(self, func):
        wraps(func)(self)
        self.ncalls = 0

    def __call__(self, *args, **kwargs):
        self.ncalls += 1
        return self.__wrapped__(*args, **kwargs)

    def __get__(self, instance, cls):
        if instance is None:
            return self
        else:
            return types.MethodType(self, instance)


class Route(dict):
    """路由, 通过函数名访问函数"""

    def __call__(self, f):
        if isfunction(f):
            self[f.__name__] = f
            return f
        else:
            raise TypeError('Route argument must be function')


def typeassert(*ty_args, **ty_kwargs):
    """类型强制检查"""
    def decorate(func):
        if not __debug__:
            return func
        sig = signature(func)
        bound_types = sig.bind_partial(*ty_args, **ty_kwargs).arguments

        @wraps(func)
        def wrapper(*args, **kwargs):
            bound_values = sig.bind(*args, **kwargs)
            for name, value in bound_values.arguments.items():
                if not isinstance(value, bound_types[name]):
                    raise TypeError('Argument {} must be {}'.format(
                        name, bound_types[name]))
            return func(*args, **kwargs)
        return wrapper
    return decorate


def debugger(func, remote=("192.168.1.178", 13333)):
    """Pycharm快速调试"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        import pydevd
        try:
            ip, port = remote
            pydevd.settrace(ip, port=port, stderrToServer=True,
                            stdoutToServer=True)
            result = func(*args, **kwargs)
            return result
        finally:
            pydevd.stoptrace()
    return wrapper


def timethis(func):
    """函数消耗时间"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print('{}\t{}'.format(func.__name__, end - start))
        return result
    return wrapper


def unquotedata(data):
    """去URL编码"""
    d = {}
    for k, v in data.items():
        try:
            d[k] = unquote(v)
        except Exception:
            d[k] = v
    return d


def gen_drop(drop_id, count, can_repeat, exclude_list):
    '''随机抽奖'''
    try:
        drop_info = copy.deepcopy(DROP[drop_id])
        drop_items = drop_info.Items
        rate_total = 0
        if exclude_list is not None:
            for i in exclude_list:
                drop_items = filter(lambda x: x.get("id") != i.get("id") or x.get("count") != i.get("count"), drop_items)

        for k in drop_items:
            rate_total += k["rate"]
        # rate_total必须-1  否则有1/(rate+1)的概率取不到物品
        rate_total -= 1
        drop_out_item = []
        random_result = 0

        for i in range(count):
            if rate_total > 0:
                random_result = random.randint(0, rate_total)
            rate_base = 0
            # python循环用倒序可以变更list
            for j in range(len(drop_items) - 1, -1, -1):
                if random_result < rate_base + drop_items[j]["rate"]:
                    new_drop_info = DROP.get(drop_items[j]["id"])
                    tmp_count = drop_items[j].get('count', 1)
                    if new_drop_info is not None:
                        new_drop_item = gen_drop(new_drop_info.Id, tmp_count, can_repeat, exclude_list)
                        for item in new_drop_item:
                            drop_out_item.append(item)
                    else:
                        tmp = {"id": drop_items[j]["id"], "count": drop_items[j]["count"]}
                        if 'package' in drop_items[j]:
                            tmp['package'] = drop_items[j]['package']
                        drop_out_item.append(tmp)

                    if can_repeat is False:
                        # 从当前列表里面移除掉这个物品 并把总的机率减低
                        rate_total -= drop_items[j]["rate"]
                        del drop_items[j]
                    break
                else:
                    rate_base += drop_items[j]["rate"]
        return drop_out_item
    except Exception:
        traceback.print_exc()


if __name__ == '__main__':
    pass
