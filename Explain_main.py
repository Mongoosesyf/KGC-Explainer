import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.sampling import sample_neighbors
from link import LinkPredict
from ReadDataset import MyMedicalDataset
from ExplainerModel import NodeExplainerModule
from KGCmodel_TuckER import *
import os
import sys

temp = sys.stdout
# f = open('screenshot_multi_KGCmodel.txt', 'w')
# sys.stdout = f

# os.environ["CUDA_VISIBLE_DEVICES"] = "5"

with torch.autograd.set_detect_anomaly(True):
    def FindSubTriplets(_src, _dst, triple_num):
        triples_file = open('/workspace1/syf/code/MyKGCExplainer/triples.txt', encoding='gbk')
        all_triples = triples_file.readlines()
        entities_dict = open('/workspace1/syf/code/MyKGCExplainer/MedicalData/medical/entities.dict', encoding='gbk')
        rel_dict = {'HasSymptom': 0, 'HasAccompany': 1, 'WithCategory': 2, 'BelongsTo': 3, 'HasCommonDrug': 4,
                    'HasRecommendDrug': 5, 'NotEat': 6, 'DoEat': 7, 'RecommendEat': 8, 'CheckIn': 9,
                    'IsOneCategoryOf': 10}
        entities = []
        sub_g_mask = [False for n in range(triple_num)]
        mask_idx_list = []
        triple_list = []
        for line in entities_dict:
            line = line.strip('\n')
            entities.append(line.split('\t')[1])
        print("src dst长度: ", len(_src))
        print("sub g mask构建中, 请稍等...")
        relation_list = []
        for i in range(len(_src)):
            if i % 500 == 0:
                print(i)
            head_name = entities[_src[i]].strip()
            tail_name = entities[_dst[i]].strip()
            triple_idx = 0
            for line in all_triples:
                line = line.strip('\n')
                sro_list = line.split('\t')
                if sro_list[0].strip() == head_name and sro_list[2].strip() == tail_name:
                    mask_idx_list.append(triple_idx)
                    trip = sro_list[0].strip() + ", " + sro_list[2].strip()
                    triple_list.append(trip)
                    relation_list.append(rel_dict.get(sro_list[1]))

                    break
                triple_idx = triple_idx + 1

        print("true的mask个数: ", len(mask_idx_list))
        mask_idx_list = list(set(mask_idx_list))
        print("去重后true的mask个数: ", len(mask_idx_list))
        for mask_idx in mask_idx_list:
            sub_g_mask[mask_idx] = True
        print("mask_idx_list: ", mask_idx_list)
        print("triple_list: ", triple_list)
        relation_tensor = torch.tensor(relation_list)

        return sub_g_mask, mask_idx_list, relation_tensor


    def FindSharedNeighbor(head, tail, graph):
        head_sub_g_in = sample_neighbors(graph, head, -1)
        head_sub_g_out = sample_neighbors(graph, head, -1, edge_dir='out')
        tail_sub_g_in = sample_neighbors(graph, tail, -1)
        tail_sub_g_out = sample_neighbors(graph, tail, -1, edge_dir='out')
        head_in, _ = head_sub_g_in.edges()
        _, head_out = head_sub_g_out.edges()
        tail_in, _ = tail_sub_g_in.edges()
        _, tail_out = tail_sub_g_out.edges()
        head_neighbor = torch.cat([head_in, head_out])
        head_neighbor = head_neighbor.tolist()
        tail_neighbor = torch.cat([tail_in, tail_out])
        tail_neighbor = tail_neighbor.tolist()
        shared_neighbor_list = []
        for i in range(len(head_neighbor)):
            if head_neighbor[i] in tail_neighbor:
                shared_neighbor_list.append(head_neighbor[i])
        print("shared neighbor数量: ", len(shared_neighbor_list))

        if len(shared_neighbor_list) > 50:
            shared_neighbor_list = shared_neighbor_list[:30]

        return shared_neighbor_list


    def FindSimilarNeighbor(head, tail, graph):
        print("进入FindSimilarNeighbor函数.")
        head_sub_g_in = sample_neighbors(graph, head, -1)
        print("head_sub_g_in采集完成.")
        head_sub_g_out = sample_neighbors(graph, head, -1, edge_dir='out')
        print("head_sub_g_out采集完成.")
        tail_sub_g_in = sample_neighbors(graph, tail, -1)
        print("tail_sub_g_in采集完成.")
        tail_sub_g_out = sample_neighbors(graph, tail, -1, edge_dir='out')
        print("tail_sub_g_out采集完成.")
        head_in, _ = head_sub_g_in.edges()
        _, head_out = head_sub_g_out.edges()
        tail_in, _ = tail_sub_g_in.edges()
        _, tail_out = tail_sub_g_out.edges()
        head_neighbor = torch.cat([head_in, head_out])
        head_neighbor = head_neighbor.tolist()
        tail_neighbor = torch.cat([tail_in, tail_out])
        tail_neighbor = tail_neighbor.tolist()
        print("头结点邻居数: ", len(head_neighbor))
        print("尾结点邻居数: ", len(tail_neighbor))

        loss_fn = nn.MSELoss()
        similarity_list = []
        print("进入获取similar neighbor emb的循环...")
        head_idx = 0
        for head_n in head_neighbor:
            head_idx += 1
            if head_idx % 50 == 0:
                print("头结点邻居已访问个数: ", head_idx)
            for tail_n in tail_neighbor:
                if head_n == tail_n:
                    break
                head_file = "/workspace1/syf/code/MyKGCExplainer/MedicalData/text_emb_result/entity" + str(head_n) + ".pt"
                tail_file = "/workspace1/syf/code/MyKGCExplainer/MedicalData/text_emb_result/entity" + str(tail_n) + ".pt"
                head_neighbor_emb = torch.load(head_file)
                tail_neighbor_emb = torch.load(tail_file)
                temp_list = [loss_fn(head_neighbor_emb, tail_neighbor_emb), head_n, tail_n]
                similarity_list.append(temp_list)
        print("完成similar neighbor的维护.")
        similarity_list.sort(key=lambda x: x[0])
        similarity_list = similarity_list[:30]
        similar_neighbor_list = []
        for _tup in similarity_list:
            similar_neighbor_list.append(_tup[1])
            similar_neighbor_list.append(_tup[2])
        return similar_neighbor_list


    def UniqueSrcDst(ori_src, ori_dst, reverse=False, target_head_idx=-1, target_tail_idx=-1):
        src_dst_list = []
        for idx in range(len(ori_src)):
            src_dst_str = str(ori_src[idx].item()) + '\t' + str(ori_dst[idx].item())
            src_dst_list.append(src_dst_str)
        if reverse:
            reversed_target_sro = str(target_tail_idx) + '\t' + str(target_head_idx)
            print("待删除的reverse目标三元组: ", reversed_target_sro)
            src_dst_list.remove(reversed_target_sro)
        print("去重前src_dst_list len: ", len(src_dst_list))
        src_dst_list = list(set(src_dst_list))
        print("去重后src_dst_list len: ", len(src_dst_list))
        ori_src = []
        ori_dst = []
        for src_dst in src_dst_list:
            src_dst = src_dst.split('\t')
            ori_src.append(int(src_dst[0]))
            ori_dst.append(int(src_dst[1]))
        ori_src = torch.tensor(ori_src)
        ori_dst = torch.tensor(ori_dst)
        return ori_src, ori_dst


    def extract_subgraph(graph, head_node, tail_node, hops=1):
        shared_neighbor_list = FindSharedNeighbor(head_node, tail_node, graph)
        similar_neighbor_list = FindSimilarNeighbor(head_node, tail_node, graph)
        seeds = torch.cat([head_node, tail_node])
        i_hop_in = sample_neighbors(graph, seeds, -1)
        i_hop_out = sample_neighbors(graph, seeds, -1, edge_dir='out')
        ori_src_in, ori_dst_in = i_hop_in.edges()
        ori_src_out, ori_dst_out = i_hop_out.edges()
        src_in_ori_g = torch.cat([ori_src_in, ori_src_out])
        dst_in_ori_g = torch.cat([ori_dst_in, ori_dst_out])

        src_in_ori_g, dst_in_ori_g = UniqueSrcDst(src_in_ori_g, dst_in_ori_g)
        retain_edge_mask, true_mask_idx_list, sub_rel = FindSubTriplets(src_in_ori_g, dst_in_ori_g, graph.num_edges())
        ori_dst = dst_in_ori_g.clone()
        for i in range(len(src_in_ori_g)):
            if src_in_ori_g[i].item() == head_node.item() or src_in_ori_g[i].item() == tail_node.item():
                if dst_in_ori_g[i].item() != head_node.item() and dst_in_ori_g[
                    i].item() != tail_node.item():
                    ori_h_node = ori_src[i].item()
                    ori_src[i] = ori_dst[i]
                    ori_dst[i] = ori_h_node
        print("reverse后sec dst长度: ", len(ori_src), len(ori_dst))
        
        # Find share neighbor
        shared_neighbor_weight = [0 for _ in range(len(sub_rel))]
        similar_neighbor_weight = [0 for _ in range(len(sub_rel))]
        shared_neighbor_array = [-1 for _ in
                                 range(len(sub_rel))]
        similar_neighbor_array = [-1 for _ in
                                  range(len(sub_rel))]
        shared_neighbor_triplets_list = []
        entities_dict = open('/workspace1/syf/code/MyKGCExplainer/MedicalData/medical/entities.dict', encoding='gbk')
        relations_dict = open('/workspace1/syf/code/MyKGCExplainer/MedicalData/medical/relations.dict', encoding='gbk')
        all_entities = entities_dict.readlines()
        all_relations = relations_dict.readlines()

        for shared_neighbor in shared_neighbor_list:
            cnt_shared = 0
            trip_head_array = []
            trip_tail_array = []
            for i in range(len(sub_rel)):
                if (src_in_ori_g[i] == shared_neighbor and dst_in_ori_g[i] == head_node.item()) \
                        or (src_in_ori_g[i] == head_node.item() and dst_in_ori_g[i] == shared_neighbor):
                    trip_head_array.append(i)
                    shared_neighbor_weight[i] = 1
                elif (src_in_ori_g[i] == shared_neighbor and dst_in_ori_g[i] == tail_node.item()) \
                        or (src_in_ori_g[i] == tail_node.item() and dst_in_ori_g[i] == shared_neighbor):
                    trip_tail_array.append(i)
                    shared_neighbor_weight[i] = 1
            print("trip_head_array长度: ", len(trip_head_array), trip_head_array)
            print("trip_tail_array长度: ", len(trip_tail_array), trip_tail_array)
            min_len = len(trip_tail_array)
            len2 = len(trip_head_array)
            if len2 < min_len:
                min_len = len2
            for array_idx in range(min_len):
                head_num = trip_head_array[array_idx]
                tail_num = trip_tail_array[array_idx]
                shared_neighbor_array[head_num] = tail_num
                shared_neighbor_array[tail_num] = head_num
        shared_neighbor_weight = torch.tensor(shared_neighbor_weight)
        cnt_similar = 0
        last_i = -1
        for similar_neighbor in similar_neighbor_list:
            print(cnt_similar, similar_neighbor)
            similar_neighbor_tensor = torch.tensor([similar_neighbor])
            for i in range(len(sub_rel)):
                if (src_in_ori_g[i] == similar_neighbor and dst_in_ori_g[i] == head_node.item()) \
                        or (src_in_ori_g[i] == head_node.item() and dst_in_ori_g[i] == similar_neighbor) \
                        or (src_in_ori_g[i] == similar_neighbor and dst_in_ori_g[i] == tail_node.item()) \
                        or (src_in_ori_g[i] == tail_node.item() and dst_in_ori_g[i] == similar_neighbor):
                    similar_neighbor_weight[i] = 1
                    break
            if cnt_similar % 2 != 0:
                similar_neighbor_array[i] = last_i
                similar_neighbor_array[last_i] = i
            cnt_similar += 1
            last_i = i
        similar_neighbor_weight = torch.tensor(similar_neighbor_weight)


        # shrink
        edge_all = torch.cat([ori_src, ori_dst])
        origin_nodes, new_edges_all = torch.unique(edge_all, return_inverse=True)
        print("origin_node长度: ", origin_nodes.size())

        n = int(new_edges_all.shape[0] / 2)
        new_src = new_edges_all[:n]
        new_dst = new_edges_all[n:]
        sub_graph = dgl.graph((new_src, new_dst), num_nodes=len(origin_nodes))
        sub_graph.edata[dgl.ETYPE] = sub_rel
        ori_node_list = origin_nodes
        ori_node_list = ori_node_list.tolist()
        ori_head_idx = head_node.item()
        ori_tail_idx = tail_node.item()
        head_idx = ori_node_list.index(ori_head_idx)
        tail_idx = ori_node_list.index(ori_tail_idx)

        target_triplet_idx = -1
        for i_trip in range(len(sub_rel)):
            if new_src[i_trip] == head_idx and new_dst[i_trip] == tail_idx and sub_rel[i_trip] == 1:
                target_triplet_idx = i_trip
                break
        print("idx if to-be-explainer-trip in new subg: ", target_triplet_idx)
        
        return sub_graph, origin_nodes, head_idx, tail_idx, target_triplet_idx, src_in_ori_g, dst_in_ori_g, sub_rel, \
               shared_neighbor_weight, similar_neighbor_weight, shared_neighbor_array, similar_neighbor_array

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Demo of GNN explainer in DGL')
    parser.add_argument('--target_rel_id', type=int, default=1)
    parser.add_argument('--target_num', type=int, default=56)
    parser.add_argument('--sub_g_hop', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=100, help='The number of epochs.')
    
    parser.add_argument('--lr', type=float, default=0.01, help='The learning rate.')
    parser.add_argument('--wd', type=float, default=0.0, help='Weight decay.')
    parser.add_argument('--h_dim', type=int, default=500, help='hidden dimension')
    parser.add_argument('--model_ckpt_path', type=str,
                        default='/workspace1/syf/code/MyKGCExplainer/model_state_only_use_nids_epochs3000_hdim500_bdd.pth')
    parser.add_argument('--result_node_num', type=int, default=10)
    parser.add_argument('--result_triple_path', type=str,
                        default="result.txt")
    args = parser.parse_args()

    data = MyMedicalDataset(raw_dir='/workspace1/syf/code/MyKGCExplainer/MedicalData/', reverse=False)
    g = data[0]
    print("g edges info: ", g.edata)
    num_nodes = g.num_nodes()
    num_rels = data.num_rels

    Internal_filename = "/workspace1/syf/code/MyKGCExplainer/new_medical_code/0618SiftedAccompany.txt"
    internal_num_list = []
    Internal_file = open(Internal_filename, encoding='utf-8')
    all_internals = Internal_file.readlines()
    for line in all_internals:
        line_list = line.split('\t')
        num = int(line_list[0])
        internal_num_list.append(num)

    # internal_num_list = [56]
    src, dst = g.edges()
    triplets = torch.stack([src, g.edata['etype'], dst], dim=1)
    test_mask = g.edata['test_mask']
    test_triples = triplets[test_mask]
    target_num = args.target_num 
    accompany_num = 0
    start_accompany_num = 0
    save_file_name = "/workspace1/syf/code/MyKGCExplainer/new_medical_code/0719weight/epoch20_top10_001.txt"
    save_file_name = "/workspace1/syf/code/MyKGCExplainer/new_medical_code/0715_AnnotationAnalysis/1015_RQ6/num56_top10_epoch100.txt"
    for t in test_triples:
        if t[1] == args.target_rel_id:
            all_zero_flag = 0
            accompany_num += 1
            if accompany_num in internal_num_list:
                print("\ncurrent accompany_num: ", accompany_num)
                print("\ntrip to be explained: ", t)
                head_entity_id = t[0]
                tail_entity_id = t[2]
                # break
                head_node_idx = torch.tensor([head_entity_id])
                tail_node_idx = torch.tensor([tail_entity_id])
                sub_g, sub_nids, new_head_idx, new_tail_idx, target_triplet_idx, src_in_ori_g, dst_in_ori_g, sub_rel, \
                shared_neighbor_weight, similar_neighbor_weight, shared_neighbor_array, similar_neighbor_array = \
                    extract_subgraph(g, head_node_idx, tail_node_idx, hops=args.sub_g_hop)
                # # 0521添加,不考虑没有shared的,因为计算指标时会/0
                if shared_neighbor_weight.eq(1).sum().item() == 0:
                    with open(save_file_name, "a") as f:
                        line_to_write = str(accompany_num) + '\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\n'
                        f.write(line_to_write)
                    continue
                sub_g.edata['norm'] = dgl.norm_by_dst(sub_g).unsqueeze(-1)
                sub_num_nodes = sub_g.num_nodes()

                # Initialize explainer
                explainer = NodeExplainerModule(num_edges=sub_g.number_of_edges())
                explainer.train()
                optim = torch.optim.Adam(explainer.parameters(), lr=args.lr, weight_decay=args.wd)

                shared_all = 0
                similar_all = 0
                KGC_score_all = 0
                new_src, new_dst = sub_g.edges()

                KGC_path_weight = [0 for _ in range(len(sub_rel))]
                head_neighbors_tups = []
                tail_neighbors_tups = []
                mult_tup = []

                # ---- rgcn----------------------------
                model = LinkPredict(num_nodes, num_rels,
                                    h_dim=args.h_dim)
                model = model.cpu()
                model_state_file = "/workspace1/syf/code/MyKGCExplainer/model_state_only_use_nids_epochs3000_hdim500_bdd.pth"
                checkpoint = torch.load(model_state_file)
                model.load_state_dict(checkpoint['state_dict'])
                model.requires_grad_(False)
                model.eval()
                print("\n开始进行model embedding.")
                model_embd = model(sub_g, sub_nids)
                print("完成model embedding.")


                for i in range(len(new_src)):

                    temp_head_idx = new_src[i].item()
                    temp_tail_idx = new_dst[i].item()

                    emb_r = model.w_relation[sub_rel[i]]
                    temp_head_idx = new_src[i].item()
                    temp_tail_idx = new_dst[i].item()
                    KGC_score = model_embd[temp_head_idx] * emb_r * model_embd[temp_tail_idx]
                    KGC_score = torch.sigmoid(torch.sum(KGC_score))

                    tup_score_idx = [KGC_score, i]
                    if temp_head_idx == new_head_idx or temp_tail_idx == new_head_idx:
                        head_neighbors_tups.append(tup_score_idx)
                    elif temp_head_idx == new_tail_idx or temp_tail_idx == new_tail_idx:
                        tail_neighbors_tups.append(tup_score_idx)
                for h_tup in head_neighbors_tups:
                    for t_tup in tail_neighbors_tups:
                        score_mult = h_tup[0] * t_tup[0]
                        temp_tup = [score_mult, h_tup[1], t_tup[1]]
                        mult_tup.append(temp_tup)
                mult_tup.sort(key=lambda x: x[0], reverse=True)
                print("mult_tup score rank: ", mult_tup)
                if len(mult_tup) <= 3:
                    with open(save_file_name, "a") as f:
                        line_to_write = str(accompany_num) + '\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\n'
                        f.write(line_to_write)
                    continue
                retain_num = 30
                KGC_path_array = [-1 for _ in range(len(sub_rel))]
                for i in range(retain_num):
                    KGC_path_weight[mult_tup[i][1]] = 1 
                    KGC_path_weight[mult_tup[i][2]] = 1
                    KGC_path_array[mult_tup[i][1]] = mult_tup[i][
                        2] 
                    KGC_path_array[mult_tup[i][2]] = mult_tup[i][1]
                print("KGC_path_weight: ", KGC_path_weight)
                KGC_score_weight = KGC_path_weight
                KGC_score_weight = torch.tensor(KGC_score_weight)

                # Train explainer
                for epoch in range(args.epochs):
                    print("\nepoch=", epoch + 1)
                    edge_mask = explainer()  # explainer forward
                    loss = explainer._loss(shared_neighbor_weight, similar_neighbor_weight,
                                           KGC_score_weight)

                    # loss backward
                    print("loss=", loss)
                    optim.zero_grad()
                    loss.backward()
                    optim.step()

                final_edge_mask = []
                for i in range(len(explainer.edge_mask)):
                    final_edge_mask.append(explainer.edge_mask.sigmoid()[i])
                rank_list = []
                for i in range(len(final_edge_mask)):
                    _tuple = [i, final_edge_mask[i]]  # belike [0, 0.673]; [1, 0.748]..
                    rank_list.append(_tuple)
                rank_list.sort(key=lambda x: x[1], reverse=True)
                # print("rank_list: ", rank_list)
                chosen_triple_list = []
                triples_file = open('/workspace1/syf/code/MyKGCExplainer/triples.txt',
                                    encoding='gbk')
                all_triples = triples_file.readlines()

                # write into result file
                result_triple_path = args.result_triple_path
                result_triple_path = "/workspace1/syf/code/MyKGCExplainer/new_medical_code/0719weight/top10_epoch50_111_/explainer_result_num_" + str(accompany_num) + ".txt"
                unpaired_result_triple_path = "/workspace1/syf/code/MyKGCExplainer/new_medical_code/0719weight/unpaired_top10_epoch50_111_/explainer_result_num_" + str(accompany_num) + ".txt"
                result_triple_path = "/workspace1/syf/code/MyKGCExplainer/new_medical_code/0715_AnnotationAnalysis/1015_RQ6/paired_result.txt"
                unpaired_result_triple_path = "/workspace1/syf/code/MyKGCExplainer/new_medical_code/0715_AnnotationAnalysis/1015_RQ6/unpaired_result.txt"
                
                entities_dict = open('/workspace1/syf/code/MyKGCExplainer/MedicalData/medical/entities.dict', encoding='gbk')
                relations_dict = open('/workspace1/syf/code/MyKGCExplainer/MedicalData/medical/relations.dict', encoding='gbk')
                all_entities = entities_dict.readlines()
                all_relations = relations_dict.readlines()

                idx_list = []
                idx_list_without_pairs = []
                _result_node_num = args.result_node_num
                if len(rank_list) < args.result_node_num:
                    _result_node_num = len(rank_list)
                for i in range(_result_node_num):
                    idx = rank_list[i][0]
                    idx_list.append(idx)
                    idx_list_without_pairs.append(idx)
                    if shared_neighbor_weight[idx] != 0:
                        idx_list.append(shared_neighbor_array[idx])

                        head_nid = src_in_ori_g[idx]
                        tail_nid = dst_in_ori_g[idx]
                        rel_id = sub_rel[idx]
                        head_name = all_entities[head_nid].split('\t')[1].strip()
                        rel_name = all_relations[rel_id].split('\t')[1].strip()
                        tail_name = all_entities[tail_nid].split('\t')[1]
                        triple_to_write1 = head_name + '\t' + rel_name + '\t' + tail_name

                        head_nid = src_in_ori_g[shared_neighbor_array[idx]]
                        tail_nid = dst_in_ori_g[shared_neighbor_array[idx]]
                        rel_id = sub_rel[shared_neighbor_array[idx]]
                        head_name = all_entities[head_nid].split('\t')[1].strip()
                        rel_name = all_relations[rel_id].split('\t')[1].strip()
                        tail_name = all_entities[tail_nid].split('\t')[1]
                        triple_to_write2 = head_name + '\t' + rel_name + '\t' + tail_name

                    if similar_neighbor_weight[idx] != 0:
                        idx_list.append(similar_neighbor_array[idx])

                        head_nid = src_in_ori_g[idx]
                        tail_nid = dst_in_ori_g[idx]
                        rel_id = sub_rel[idx]
                        head_name = all_entities[head_nid].split('\t')[1].strip()
                        rel_name = all_relations[rel_id].split('\t')[1].strip()
                        tail_name = all_entities[tail_nid].split('\t')[1]
                        triple_to_write1 = head_name + '\t' + rel_name + '\t' + tail_name

                        head_nid = src_in_ori_g[similar_neighbor_array[idx]]
                        tail_nid = dst_in_ori_g[similar_neighbor_array[idx]]
                        rel_id = sub_rel[similar_neighbor_array[idx]]
                        head_name = all_entities[head_nid].split('\t')[1].strip()
                        rel_name = all_relations[rel_id].split('\t')[1].strip()
                        tail_name = all_entities[tail_nid].split('\t')[1]
                        triple_to_write2 = head_name + '\t' + rel_name + '\t' + tail_name

                    if KGC_score_weight[idx] != 0:
                        idx_list.append(
                            KGC_path_array[idx])

                        head_nid = src_in_ori_g[idx]
                        tail_nid = dst_in_ori_g[idx]
                        rel_id = sub_rel[idx]
                        head_name = all_entities[head_nid].split('\t')[1].strip()
                        rel_name = all_relations[rel_id].split('\t')[1].strip()
                        tail_name = all_entities[tail_nid].split('\t')[1]
                        triple_to_write1 = head_name + '\t' + rel_name + '\t' + tail_name

                        head_nid = src_in_ori_g[KGC_path_array[idx]]
                        tail_nid = dst_in_ori_g[KGC_path_array[idx]]
                        rel_id = sub_rel[KGC_path_array[idx]]
                        head_name = all_entities[head_nid].split('\t')[1].strip()
                        rel_name = all_relations[rel_id].split('\t')[1].strip()
                        tail_name = all_entities[tail_nid].split('\t')[1]
                        triple_to_write2 = head_name + '\t' + rel_name + '\t' + tail_name

                idx_list = list(set(idx_list))
                idx_list_without_pairs = list(set(idx_list_without_pairs))

                result_triplets_list = []
                result_triplets_list_without_pair = []
                TP_shared = 0
                TP_similar = 0
                TP_KGC_score = 0

                TP_shared_without_pair = 0
                TP_similar_without_pair = 0
                TP_KGC_score_without_pair = 0

                FP_overall = 0
                for idx in idx_list:
                    triple_type = 4  # Nothing
                    if KGC_score_weight[idx] != 0:
                        triple_type = 3  # path
                        TP_KGC_score += 1
                    if similar_neighbor_weight[idx] != 0:
                        triple_type = 2  # similar
                        TP_similar += 1
                    if shared_neighbor_weight[idx] != 0:
                        triple_type = 1  # shared
                        TP_shared += 1
                    if triple_type == 4:
                        FP_overall += 1
                    head_nid = src_in_ori_g[idx]
                    tail_nid = dst_in_ori_g[idx]
                    rel_id = sub_rel[idx]
                    head_name = all_entities[head_nid].split('\t')[1].strip()
                    rel_name = all_relations[rel_id].split('\t')[1].strip()
                    tail_name = all_entities[tail_nid].split('\t')[1].strip()
                    triple_to_write = head_name + '\t' + rel_name + '\t' + tail_name + '\t' + str(triple_type) + '\n'
                    result_triplets_list.append(triple_to_write)

                FP_overall_without_pair = 0
                for idx in idx_list_without_pairs:
                    triple_type = 4
                    if KGC_score_weight[idx] != 0:
                        triple_type = 3
                        TP_KGC_score_without_pair += 1
                    if similar_neighbor_weight[idx] != 0:
                        triple_type = 2
                        TP_similar_without_pair += 1
                    if shared_neighbor_weight[idx] != 0:
                        triple_type = 1
                        TP_shared_without_pair += 1
                    if triple_type == 4:
                        FP_overall_without_pair += 1
                    head_nid = src_in_ori_g[idx]
                    tail_nid = dst_in_ori_g[idx]
                    rel_id = sub_rel[idx]
                    head_name = all_entities[head_nid].split('\t')[1].strip()
                    rel_name = all_relations[rel_id].split('\t')[1].strip()
                    tail_name = all_entities[tail_nid].split('\t')[1].strip()
                    triple_to_write = head_name + '\t' + rel_name + '\t' + tail_name + '\t' + str(triple_type) + '\n'
                    result_triplets_list_without_pair.append(triple_to_write)

                head_name = all_entities[head_entity_id].split('\t')[1].strip()
                rel_name = all_relations[args.target_rel_id].split('\t')[1].strip()
                tail_name = all_entities[tail_entity_id].split('\t')[1].strip()
                triple_to_write = head_name + '\t' + rel_name + '\t' + tail_name + '\t0\n'
                result_triplets_list.append(triple_to_write)
                result_triplets_list_without_pair.append(triple_to_write)

                # 写入
                result_triplets_list = list(set(result_triplets_list))
                with open(result_triple_path, "a") as f:
                    for trip in result_triplets_list:
                        f.write(trip)
                print("paired subg triples finished writting.")

                result_triplets_list_without_pair = list(set(result_triplets_list_without_pair))
                with open(unpaired_result_triple_path, "a") as f:
                    for trip in result_triplets_list_without_pair:
                        f.write(trip)
                print("unpaired subg triples finished writting.")
                
                f.close()

                # calculate evaluation metrics
                total_num = len(result_triplets_list)
                total_num_without_pair = len(result_triplets_list_without_pair)
                shared_neighbor_num = shared_neighbor_weight.eq(1).sum().item()
                similar_neighbor_num = similar_neighbor_weight.eq(1).sum().item()
                KGC_score_num = KGC_score_weight.eq(1).sum().item()
                overall_weight = shared_neighbor_weight + similar_neighbor_weight + KGC_score_weight
                overall_num = torch.count_nonzero(overall_weight)
                FN_shared = shared_neighbor_num - TP_shared
                FN_similar = similar_neighbor_num - TP_similar
                FN_KGC_score = KGC_score_num - TP_KGC_score_without_pair
                
                # without pair
                FN_shared_without_pair = shared_neighbor_num - TP_shared_without_pair
                FN_similar_without_pair = similar_neighbor_num - TP_similar_without_pair
                FN_KGC_score_without_pair = KGC_score_num - TP_KGC_score_without_pair

                recall_shared = TP_shared / (TP_shared + FN_shared)
                recall_similar = TP_similar / (TP_similar + FN_similar)
                recall_KGC_score = TP_KGC_score / (TP_KGC_score + FN_KGC_score)
                recall_overall = (total_num - FP_overall) / overall_num
                recall_overall = float(recall_overall.item())

                recall_shared_without_pair = TP_shared_without_pair / (TP_shared_without_pair + FN_shared_without_pair)
                recall_similar_without_pair = TP_similar_without_pair / (
                            TP_similar_without_pair + FN_similar_without_pair)
                recall_KGC_score_without_pair = TP_KGC_score_without_pair / (
                            TP_KGC_score_without_pair + FN_KGC_score_without_pair)
                recall_overall_without_pair = (total_num_without_pair - FP_overall_without_pair) / overall_num
                recall_overall_without_pair = float(recall_overall_without_pair.item())

                # precision
                precision_shared = TP_shared / total_num  # TP / (TP+FP)
                precision_similar = TP_similar / total_num
                precision_KGC_score = TP_KGC_score / total_num
                precision_overall = (total_num - FP_overall) / total_num

                precision_shared_without_pair = TP_shared_without_pair / total_num_without_pair
                precision_similar_without_pair = TP_similar_without_pair / total_num_without_pair
                precision_KGC_score_without_pair = TP_KGC_score_without_pair / total_num_without_pair
                precision_overall_without_pair = (total_num_without_pair - FP_overall_without_pair) / total_num_without_pair

                # ACC overall
                sub_g_size = len(sub_rel)
                all_N = sub_g_size - overall_num
                TN_overall = all_N - FP_overall
                ACC_overall = (total_num - FP_overall + TN_overall) / sub_g_size  # (TP+TN) / (TP+TN+FP+FN)
                ACC_overall = float(ACC_overall.item())
                # without pair:
                TN_overall_without_pair = all_N - FP_overall_without_pair
                ACC_overall_without_pair = (total_num_without_pair - FP_overall_without_pair + TN_overall_without_pair) / sub_g_size
                ACC_overall_without_pair = float(ACC_overall_without_pair.item())

                # percent
                total_num = len(result_triplets_list)
                num_shared = 0
                num_similar = 0
                num_KGC_score = 0
                for content in result_triplets_list:
                    content = content.strip()
                    type_num = int(content.split('\t')[-1])
                    if type_num == 1:
                        num_shared += 1
                    elif type_num == 2:
                        num_similar += 1
                    elif type_num == 3:
                        num_KGC_score += 1
                perc_shared = num_shared / total_num
                perc_similar = num_similar / total_num
                perc_KGC_score = num_KGC_score / total_num
                perc_others = 1 - perc_shared - perc_similar - perc_KGC_score

                total_num_without_pair = len(result_triplets_list_without_pair)
                num_shared_without_pair = 0
                num_similar_without_pair = 0
                num_KGC_score_without_pair = 0
                for content in result_triplets_list_without_pair:
                    content = content.strip()
                    type_num = int(content.split('\t')[-1])
                    if type_num == 1:
                        num_shared_without_pair += 1
                    elif type_num == 2:
                        num_similar_without_pair += 1
                    elif type_num == 3:
                        num_KGC_score_without_pair += 1
                perc_shared_without_pair = num_shared_without_pair / total_num_without_pair
                perc_similar_without_pair = num_similar_without_pair / total_num_without_pair
                perc_KGC_score_without_pair = num_KGC_score_without_pair / total_num_without_pair
                perc_others_without_pair = 1 - perc_shared_without_pair - perc_similar_without_pair - perc_KGC_score_without_pair

                with open(save_file_name, "a") as f:
                    rel1_num = str(accompany_num)

                    if (float(precision_shared) + float(recall_shared)) == 0 or (
                            float(precision_similar) + float(recall_similar)) == 0 or (
                            float(precision_KGC_score) + float(recall_KGC_score)) == 0 or (
                            float(precision_overall) + float(recall_overall)) == 0 or (
                            float(precision_shared_without_pair) + float(recall_shared_without_pair)) == 0 or (
                            float(precision_similar_without_pair) + float(recall_similar_without_pair)) == 0 or (
                            float(precision_KGC_score_without_pair) + float(recall_KGC_score_without_pair)) == 0 or (
                            float(precision_overall_without_pair) + float(recall_overall_without_pair)) == 0:
                        with open(save_file_name, "a") as f:
                            line_to_write = str(accompany_num) + '\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\n'
                            f.write(line_to_write)
                        continue

                    F1_shared = str((2 * float(precision_shared) * float(recall_shared)) / (
                                float(precision_shared) + float(recall_shared)))
                    F1_similar = str((2 * float(precision_similar) * float(recall_similar)) / (
                                float(precision_similar) + float(recall_similar)))
                    F1_KGC = str((2 * float(precision_KGC_score) * float(recall_KGC_score)) / (
                                float(precision_KGC_score) + float(recall_KGC_score)))
                    F1_overall = str((2 * float(precision_overall) * float(recall_overall)) / (
                                float(precision_overall) + float(recall_overall)))

                    # recall:
                    recall_shared = str(recall_shared)
                    recall_similar = str(recall_similar)
                    recall_KGC = str(recall_KGC_score)
                    recall_overall = str(recall_overall)

                    # precison:
                    precision_shared = str(precision_shared)
                    precision_similar = str(precision_similar)
                    precision_KGC = str(precision_KGC_score)
                    precision_overall = str(precision_overall)

                    # percentage:
                    per_shared = str(perc_shared)
                    per_similar = str(perc_similar)
                    per_KGC = str(perc_KGC_score)
                    per_others = str(perc_others)

                    # ACC:
                    ACC_overall = str(ACC_overall)

                    # unpaired
                    # F1:
                    F1_shared_without_pair = str(
                        (2 * float(precision_shared_without_pair) * float(recall_shared_without_pair)) / (
                                    float(precision_shared_without_pair) + float(recall_shared_without_pair)))
                    F1_similar_without_pair = str(
                        (2 * float(precision_similar_without_pair) * float(recall_similar_without_pair)) / (
                                    float(precision_similar_without_pair) + float(recall_similar_without_pair)))
                    F1_KGC_without_pair = str(
                        (2 * float(precision_KGC_score_without_pair) * float(recall_KGC_score_without_pair)) / (
                                    float(precision_KGC_score_without_pair) + float(recall_KGC_score_without_pair)))
                    F1_overall_without_pair = str(
                        (2 * float(precision_overall_without_pair) * float(recall_overall_without_pair)) / (
                                    float(precision_overall_without_pair) + float(recall_overall_without_pair)))

                    # recall:
                    recall_shared_without_pair = str(recall_shared_without_pair)
                    recall_similar_without_pair = str(recall_similar_without_pair)
                    recall_KGC_without_pair = str(recall_KGC_score_without_pair)
                    recall_overall_without_pair = str(recall_overall_without_pair)

                    # precison:
                    precision_shared_without_pair = str(precision_shared_without_pair)
                    precision_similar_without_pair = str(precision_similar_without_pair)
                    precision_KGC_without_pair = str(precision_KGC_score_without_pair)
                    precision_overall_without_pair = str(precision_overall_without_pair)

                    # percentage:
                    per_shared_without_pair = str(perc_shared_without_pair)
                    per_similar_without_pair = str(perc_similar_without_pair)
                    per_KGC_without_pair = str(perc_KGC_score_without_pair)
                    per_others_without_pair = str(perc_others_without_pair)

                    # ACC:
                    ACC_overall_without_pair = str(ACC_overall_without_pair)

                    line_to_write = rel1_num + '\t' + ACC_overall + '\t' + precision_overall + '\t' + recall_overall + '\t' + F1_overall + '\t' + \
                                    per_shared + '\t' + per_similar + '\t' + per_KGC + '\t' + per_others + '\t' + \
                                    recall_shared + '\t' + recall_similar + '\t' + recall_KGC + '\t' + \
                                    ACC_overall_without_pair + '\t' + precision_overall_without_pair + '\t' + recall_overall_without_pair + '\t' + F1_overall_without_pair + '\t' + \
                                    per_shared_without_pair + '\t' + per_similar_without_pair + '\t' + per_KGC_without_pair + '\t' + per_others_without_pair + '\t' + \
                                    recall_shared_without_pair + '\t' + recall_similar_without_pair + '\t' + recall_KGC_without_pair + '\n'
                    f.write(line_to_write)
