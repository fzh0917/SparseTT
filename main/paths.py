r"""
Get root path & root config path & 
"""
import os.path as osp

import sys  # isort:skip

module_name = "main"
p = __file__
while osp.basename(p) != module_name:
    p = osp.dirname(p)

# video_analyst root
ROOT_PATH = osp.dirname(p)
ROOT_CFG = osp.join(ROOT_PATH, 'config.yaml')
sys.path.insert(0, ROOT_PATH)  # isort:skip
