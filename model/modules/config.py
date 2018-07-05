# coding=utf-8
import sys
import os
parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
parent_parent_dir = os.path.dirname(parent_dir)
current_dir = os.getcwd()
sys.path.insert(0, parent_dir)
sys.path.insert(0, parent_parent_dir)
