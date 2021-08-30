import numpy as np

def multidict(dict_):
    n = len(dict_)
    number = list(np.arange(n)+1)
    return [number, dict_]

def convert_cost_arr(cdict):
    I = list(np.unique(np.array(list(cdict.keys()))[:,0]))
    J = list(np.unique(np.array(list(cdict.keys()))[:,1]))
    cost = np.zeros((len(J), len(I)))
    for j in J:
        for i in I:
            cost[j-1, i-1] = int(cdict[(i,j)])
    return cost

def extract_empty_cell(TP):
    ec = list()
    for i in range(TP.shape[0]):
        for j in range(TP.shape[1]):
            if TP[i,j] == 0:
                ec.append((i,j))
    return ec

def empty_labeling(arr, index=None, temp=False):
    arr_bin = arr.copy()
    arr_bin[arr_bin>0] = 1
    if np.min(arr_bin) < 0:
        print('Error')
    if temp:
        arr_bin[index[0],index[1]] = 1
    return arr_bin

def route_labeling(TP_bin_temp, index):
    sum_axis_col = np.sum(TP_bin_temp, axis=0)
    sum_axis_row = np.sum(TP_bin_temp, axis=1)
    TP_route = np.zeros_like(TP_bin_temp)
    for i,row in enumerate(sum_axis_row):
        for j,col in enumerate(sum_axis_col):
            if row > 1 and col > 1:
                if i == index[0] or j == index[1]:
                    TP_route[i, j] = -1
                else:
                    TP_route[i, j] = 1
    TP_route[index[0], index[1]] = 1
    return TP_route

def extract_route_pattern(index, TP_route):
    index_col = list()
    index_row = list()
    for i in range(TP_route.shape[0]):
            for j in range(TP_route.shape[1]):
                index_temp = (i,j)
                if index_temp:
                    if index_temp[0] == index[0]:
                        index_col.append(index_temp)
                    elif index_temp[1] == index[1]:
                        index_row.append(index_temp)
    route = list()
    for idx_row in index_row:
        for idx_col in index_col:
            corner_index = (idx_row[0],idx_col[1])
            index_set = [index, idx_row, idx_col, corner_index]
            route.append(index_set)
    return route

def extract_route_pattern_labeling(TP_route, extracted_route):
    mask = np.zeros_like(TP_route)
    for point in extracted_route:
        mask[point[0], point[1]] = 1
    TP_route_mask = (TP_route*mask).copy()
    return TP_route_mask

def create_list_route_pattern_labelings(index, TP_route):
    TP_route_masks = list()
    for extracted_route in extract_route_pattern(index, TP_route):
        TP_route_mask = extract_route_pattern_labeling(TP_route, extracted_route)
        cnt = TP_route_mask[TP_route_mask!=0].shape[0]
        TP_route_mask = TP_route_mask.tolist()
        contain = [arr==TP_route_mask for arr in TP_route_masks]
        if cnt % 2 == 0 and (not True in contain):
            TP_route_masks.append(TP_route_mask)
    return TP_route_masks

def compute_cost(TP, cost):
    return np.sum(TP*cost)

def minus_or_not(arr):
    return np.min(arr) >=0

def compute_best_params(TP_route_masks, TP, cost, unit, best_cost):
    best_params = dict()
    for TP_route_mask in TP_route_masks:
        num = 1
        while True:
            UNIT = np.zeros_like(TP)+(unit*num)
            TP_next = ((TP_route_mask*UNIT)+TP).copy()
            if not minus_or_not(TP_next):
                break
            temp_cost = compute_cost(TP_next, cost)
            if best_cost >temp_cost:
                best_cost = temp_cost
                best_params['TP_route'] = TP_route_mask
                best_params['num'] = num
                best_params['cost'] = temp_cost
            num+=1
    return best_params

# 問題設定
def init():
    # 需要量
    demand = {1:80, 2:270, 3:250, 4:160, 5:180}
    # 容量
    capacity = {1:500, 2:500, 3:500}
    # 輸送単位
    unit = 10
    # 単位あたりの輸送費用
    ckey = [(1, 1),(1, 2),(1, 3),(2, 1),(2, 2),(2, 3),(3, 1),(3, 2),(3, 3),(4, 1),(4, 2),(4, 3),(5, 1),(5, 2),(5, 3)]
    cvalue = [4, 6, 9, 5, 4, 7, 6, 3, 4, 8, 5, 3, 10, 8, 4]
    cost = convert_cost_arr(dict(zip(ckey, cvalue)))
    return demand, capacity, unit, cost

# 北西隅の方法
def north_west_corner_method(demand, capacity, unit):
    I,D = multidict(demand)
    J,M = multidict(capacity)
    TP = np.zeros((len(J), len(I)))
    for j in J:
        for i in I:
            if M[j] >= unit and D[i] >= unit:
                tp = 0
                m = int(M[j])
                d = int(D[i])
                while True:
                    tp += unit
                    if tp == m or tp == d:
                        m_sub = m - tp
                        d_sub = d - tp
                        tp_set = tp
                        break
                    elif tp > m or tp > d:
                        m_sub = m - (tp - unit)
                        d_sub = d - (tp - unit)
                        tp_set = (tp - unit)
                        break
                M[j] = m_sub
                D[i] = d_sub
                TP[j-1,i-1] = tp_set
    return TP

# 飛び石法
def stepping_stone_method(TP, best_cost, cost):
    # 更新履歴
    update_history = {0:{'cost':best_cost,'TP':TP}}
    # 更新部分
    best_cost_main = best_cost
    update = int()
    while True:
        update+=1
        # 1.空白セルの列挙
        ec = extract_empty_cell(TP)
        for i,index in enumerate(ec):
            # 2.輸送量変更の全経路を抽出
            TP_bin = empty_labeling(TP)
            TP_bin_temp = empty_labeling(TP, index, temp=True)
            TP_route = route_labeling(TP_bin_temp, index)
            TP_route_masks = create_list_route_pattern_labelings(index, TP_route)
            # 3.輸送コストが最小となる（追加経路×輸送量）の組み合わせを抽出
            best_params = compute_best_params(TP_route_masks, TP, cost, unit, best_cost=best_cost_main)
            if len(best_params) > 0:
                best_cost = best_params['cost']
            else:
                best_cost = best_cost_main
            if i == 0:
                best_cost_next = best_cost
                best_params_next = best_params
            else:
                if best_cost_next > best_cost:
                    best_cost_next = best_cost
                    best_params_next = best_params
        # 4.輸送量の更新
        if len(best_params_next) == 0:
            break
        UNIT = np.zeros_like(TP) + (unit * best_params_next['num'])
        TP = (TP + (best_params_next['TP_route'] * UNIT)).copy()
        #print(best_cost_main, best_cost_next)
        if best_cost_main > best_cost_next:
            best_cost_main = best_cost_next
            TP_best = TP
        else:
            break
        update_history[update] = {'cost':best_cost_next,'TP':TP}
    return TP_best, best_cost_main, update_history


if __name__ == '__main__':
    # 問題設定
    demand, capacity, unit, cost = init()
    # 北西隅の方法
    TP = north_west_corner_method(demand, capacity, unit)
    cost_init = compute_cost(TP, cost)
    # 飛び石法
    TP_best, best_cost, update_history = stepping_stone_method(TP, cost_init, cost)
    print(update_history)