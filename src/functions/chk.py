# -*- coding: utf-8 -*-
"""
chk - This is a module realizing perfect print functions for debugging.
"""
import inspect
import sys
import numpy as np


def chkprint(*args):
    """
    This is a perfect print function for debugging .
    let's try '$ python chk.py' on your console.
    """
    of = inspect.getouterframes(sys._getframe())
    # (0: current frame, 1: outer frame), (2: lineno, 3: function, 4:code)
    _, lineno, code = of[0][3], str(of[1][2]), str(of[1][4])[2:-2]
    code = code[code.find("(") + 1 : code.rfind(")")]
    buf, count = "", 0
    for c in code.split(","):
        if len(buf):
            buf += ", " + c.strip()
        else:
            buf += c.strip()
        if buf.count("[") == buf.count("]") and buf.count("(") == buf.count(")"):
            print(lineno + ": " + buf + " = " + repr(args[count]))
            buf = ""
            count += 1


def chkshape(*args):
    """
    This is a numpy.ndarray.shape version of chkprint function.
    let's try '$ python chk.py' on your console.
    Requirement: NumPy
    """
    of = inspect.getouterframes(sys._getframe())
    # (0: current frame, 1: outer frame), (2: lineno, 3: function, 4:code)
    _, lineno, code = of[0][3], str(of[1][2]), str(of[1][4])[2:-2]
    code = code[code.find("(") + 1 : code.rfind(")")]
    buf, count = "", 0
    for c in code.split(","):
        if len(buf):
            buf += ", " + c.strip()
        else:
            buf += c.strip()
        if buf.count("[") == buf.count("]") and buf.count("(") == buf.count(")"):
            print(lineno + ": " + buf + ".shape = " + str(np.shape(args[count])))
            buf = ""
            count += 1


def main():
    """
    This is an entry point and a test script
    when you execute this module as a script.
    """
    # variables for test print
    a = 1
    lst = ["a", "b", "c", "d", "e"]
    hoge = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])

    class x:
        y = "hello"

    # default print
    print("---default print---")
    print(a)
    print(lst)
    print(a, lst, hoge, hoge[0, 1:4])
    print(x.y)
    print("")

    # perfect print by chkprint
    print("---perfect print---")
    chkprint(a)
    chkprint(lst)
    chkprint(a, lst, hoge, hoge[0, 1:4])
    chkprint(x.y)
    print("")

    # perfect numpy.ndarray.shape by chkshpae
    print("---for np.shape---")
    chkshape(hoge)
    chkshape(hoge, hoge[0, 1:4])


if __name__ == "__main__":
    main()
