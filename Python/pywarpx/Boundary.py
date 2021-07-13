# Copyright 2021 Roelof Groenewald
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

from .Bucket import Bucket

boundary = Bucket('boundary')
boundary_list = []

def newboundary(name):
    result = Bucket(name)
    boundary_list.append(result)
    return result
