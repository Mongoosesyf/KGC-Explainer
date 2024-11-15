# order for visualization: streamlit run D:/mycode/python/GraduationProject/SpiderTest/visualization.py
# artical: https://blog.csdn.net/itshard/article/details/126821778
# code: https://github.com/ChrisDelClea/streamlit-agraph
import streamlit
from streamlit_agraph import agraph, Node, Edge, Config
import sys
import string
import numpy as np

data_path = ''

head_name = "鼻腔癌"
tail_name = "鼻出血"

op = open('txt_for_visualization/new_Medical/1GAT.txt', 'r')
_list = []
num = 0
for line in op:
    _list.append(line)
    num += 1
subs = []
objs = []
rels = []

for i in range(0, num):
    print(i)
    subs.append(str(_list[i].split('\t')[0]))
    objs.append(str((_list[i].split('\t')[2])))
    rels.append(str((_list[i].split('\t')[3]).strip()))  # rel type

a = list(set(subs))
b = list(set(objs))
c = a + b
c = list(set(c))
nodes = []
edges = []

for i in range(0, len(c)):
    if c[i] == head_name or c[i] == tail_name:
        nodes.append(Node(id=str(c[i]),
                          label=c[i],
                          size=20,  # default=25
                          shape="circularImage",
                          image="http://i0.hdslb.com/bfs/article/aafceafc4edf590856866c4ed0d56d4d7a6c8635.jpg",
                          color="#ff0000"
                          )
                     )
    else:  # default color is light green
        nodes.append(Node(id=str(c[i]),
                          label=c[i],
                          size=20,  # 默认25
                          shape="circularImage",
                          image="http://i0.hdslb.com/bfs/article/aafceafc4edf590856866c4ed0d56d4d7a6c8635.jpg",
                          # color="#ff0000"
                          )
                     )  # includes **kwargs

for i in range(0, num):
    # print("rels[i]=", rels[i])
    color_str = "#6e6e6e"  # grey
    if rels[i] == '0':
        color_str = "#ff0000"  # trip to be explained: red
    elif rels[i] == '1':
        color_str = "#000000"  # shared neighbor, black
    elif rels[i] == '2':
        color_str = "#00bf93"  # similar neighbor, green
    elif rels[i] == '3':
        color_str = "#c89600"  # KGC path, yellow
    edges.append(Edge(source=str(subs[i]),
                      label=_list[i].split('\t')[1],
                      target=str(objs[i]),
                      # type="CURVE_SMOOTH"
                      color=color_str,
                      # **kwargs
                      )
                 )

config = Config(width=1000,
                height=1000,
                # **kwargs
                )

return_value = agraph(nodes=nodes,
                      edges=edges,
                      config=config)
