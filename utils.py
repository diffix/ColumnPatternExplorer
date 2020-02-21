import os
import re

import numpy as np


def getCharacter(chars, bias):
    return choice(chars, p=bias)


def is_valid_address(email):
    regex = r'(^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$)'
    return re.fullmatch(regex, email)


def getRandomVal(arr):
    return arr[np.random.randint(0, len(arr))]


def getRandomString(strings, probs):
    probs = np.array(probs, dtype=np.float64)
    probs /= probs.sum()
    return choice(strings, p=probs)


def choice(choices, p):
    running_sum = np.cumsum(p, dtype=np.float64)
    u = np.random.uniform(0.0, running_sum[-1])
    i = np.searchsorted(running_sum, u, side='left')
    return choices[i]


def getEnvVars(getRaw=False):
    cloak_user_env, cloak_password_env, raw_user_env, raw_password_env = None, None, None, None
    cloak_user_env = os.environ.get('CLOAK_USER')
    if cloak_user_env is None:
        print("Must set environment variable CLOAK_USER")
        quit()
    cloak_password_env = os.environ.get('CLOAK_PASS')
    if cloak_password_env is None:
        print("Must set environment variable CLOAK_PASS")
        quit()
    if getRaw:
        raw_user_env = os.environ.get('RAW_USER')
        if raw_user_env is None:
            print("Must set environment variable RAW_USER")
            quit()
        raw_password_env = os.environ.get('RAW_PASS')
        if raw_password_env is None:
            print("Must set environment variable RAW_PASS")
            quit()

    return cloak_user_env, cloak_password_env, raw_user_env, raw_password_env
