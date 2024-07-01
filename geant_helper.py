"""
read topology, save to ascii format


read traffic matrices, save to numpy array
"""
import os
# import networkx as nx
import numpy as np
import xml.etree.ElementTree as ET
from tqdm import tqdm
import pickle

def handle_topology(file='./data/topology.xml'):
    tree = ET.parse(file)
    root = tree.getroot()
    ns = root[1] # meta, netowrkStructure, demands
    nodes, links = ns[0], ns[1]
    idx_map = {}
    for i, node in enumerate(nodes):
        name = node.attrib['id']
        idx_map[name] = i

    out_file = open('./GEANT', 'w')
    for link in links:
        src, dst = link[0].text, link[1].text
        module = link[2][0]
        capacity, cost = int(float(module[0].text)), int(float(module[1].text))
        src_id, dst_id = idx_map[src], idx_map[dst]
        out_file.write(f"{src_id} {dst_id} {cost} {capacity}\n")
        out_file.write(f"{dst_id} {src_id} {cost} {capacity}\n")
    return idx_map


def handle_single_TM(file='./data/topology.xml', idx_map={}):
    tree = ET.parse(file)
    root = tree.getroot()
    demands = root[2] # meta, netowrkStructure, demands
    if len(demands) == 0:
        return None
    result = np.zeros(shape=(22, 22))
    for demand in demands:
        src = demand[0].text
        dst = demand[1].text
        val = float(demand[2].text)
        src_id, dst_id = idx_map[src], idx_map[dst]
        assert 0 <= src_id < 22 and 0 <= dst_id < 22
        result[src_id][dst_id] = val
    return result


def handle_TM(path='./data/TM/', idx_map={}):
    files = list(os.listdir(path))
    files = sorted(files)
    tms = []
    for file in tqdm(files):
        tm = handle_single_TM(path+file, idx_map)
        if tm is not None:
            tms.append(tm)
    tms = np.array(tms)
    print(tms.shape)
    pickle.dump(tms, open('GEANT.pkl', 'wb'))

if __name__ == '__main__':
    idx_map = handle_topology()
    handle_TM(idx_map=idx_map)
