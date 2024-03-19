#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------------------
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# The text of the license is located in the root directory of the source package.
# -------------------------------------------------------------------------------------

"""Useful types for argument parsing.
"""

__author__ = "Tristan Hoellinger"
__version__ = "0.1"
__date__ = "2024"
__license__ = "GPLv3"

from argparse import ArgumentParser, ArgumentTypeError


def none_or_bool_or_str(value):
    if value == "None":
        return None
    elif value == "True":
        return True
    elif value == "False":
        return False
    return value


def bool_sh(value):
    if value == "True":
        return True
    elif value == "False":
        return False
    else:
        raise ArgumentTypeError("Boolean value expected.")
