#!/usr/bin/env python3
"""
Test script to demonstrate function name logging
"""

import sys

sys.path.append("/u6/cjrisi/nocturnal/src/train")

from ttm import info_print


def test_function_1():
    """Test function that calls info_print"""
    info_print("This is from test_function_1")
    info_print("Multiple", "arguments", "from", "test_function_1")


def test_function_2():
    """Another test function"""
    info_print("This is from test_function_2")
    test_function_1()  # Call another function


def main():
    info_print("This is from main function")
    test_function_1()
    test_function_2()


if __name__ == "__main__":
    info_print("This is from module level")
    main()
