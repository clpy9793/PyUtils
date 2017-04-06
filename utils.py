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
from inspect import signature, isfunction
from functools import wraps
from urllib.parse import unquote
from traceback import format_exc


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
    try:
        d = copy.deepcopy(data)
        for k, v in d.items():
            if v and isinstance(v, str):
                d[k] = unquote(v)
        else:
            return d
    except Exception:
        return data


if __name__ == '__main__':
    pass
