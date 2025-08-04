import pandas as pd
import random
import heapq
import numpy as np

# ---------- 설정 ----------
CSV_PATH = "route.csv"
RED_REWARD = 3.0
MIN_RED_TOTAL = 5.0
POP_SIZE = 200
GENERATIONS = 300
TOURNAMENT_K = 3
CROSSOVER_PB = 0.8
MUTATION_PB = 0.3
MIN_VISIT = 8
MAX_VISIT = 8
ELITE_K = 2
FIX_START = "도서관"
# ---------------------------

def load_csv(path):
    return pd.read_csv('/content/drive/MyDrive/route.csv', encoding='cp949')

def build_graph(df):
    edges = {}       # weight graph
    red_len = {}     # red length graph
    for _, r in df.iterrows():
        a, b = r['출발점'], r['도착점']
        w = float(r['가중치']) if not pd.isna(r['가중치']) else 1e6
        red = float(r.get('빨간길(m)', 0.0)) if not pd.isna(r.get('빨간길(m)', 0.0)) else 0.0
        edges.setdefault(a, {})[b] = w
        edges.setdefault(b, {})[a] = w
        red_len.setdefault(a, {})[b] = red
        red_len.setdefault(b, {})[a] = red
    return edges, red_len

def dijkstra_with_red(start, nodes, edges, red_len):
    # 우선순위 큐: (dist, -red_accum, node)
    dist = {n: float('inf') for n in nodes}
    red_accum = {n: 0.0 for n in nodes}
    prev = {n: None for n in nodes}
    dist[start] = 0.0
    red_accum[start] = 0.0
    heap = [(0.0, 0.0, start)]  # note second is negative for maximizing red on ties
    while heap:
        d, neg_red, u = heapq.heappop(heap)
        if d > dist[u]:
            continue
        for v, w in edges.get(u, {}).items():
            cand_dist = dist[u] + w
            cand_red = red_accum[u] + red_len[u].get(v, 0.0)
            # 비교: 우선 거리 작으면 갱신, 거리 같으면 빨간길 많으면 갱신
            if (cand_dist < dist[v]) or (abs(cand_dist - dist[v]) < 1e-6 and cand_red > red_accum[v]):
                dist[v] = cand_dist
                red_accum[v] = cand_red
                prev[v] = u
                heapq.heappush(heap, (cand_dist, -cand_red, v))
    return dist, red_accum, prev  # prev는 경로 복원을 위해 필요하면 사용할 수 있음

def precompute_shortest(nodes, edges, red_len):
    n = len(nodes)
    dist_matrix = {a: {} for a in nodes}
    red_matrix = {a: {} for a in nodes}
    for a in nodes:
        dist_from_a, red_from_a, _ = dijkstra_with_red(a, nodes, edges, red_len)
        for b in nodes:
            dist_matrix[a][b] = dist_from_a[b]
            red_matrix[a][b] = red_from_a[b]
    return dist_matrix, red_matrix

# --- 기존 GA 관련 함수들, 단 edge lookup은 precomputed 사용 ---
def evaluate(individual, dist_matrix, red_matrix):
    mask, order = individual
    if FIX_START not in order:
        return float('inf'), {"reason": "missing fixed start", "num": 0}
    selected = [node for keep, node in zip(mask, order) if keep == 1]
    if FIX_START not in selected:
        return float('inf'), {"reason": f"{FIX_START} not included", "num": len(selected)}
    num = len(selected)
    if num < MIN_VISIT or num > MAX_VISIT:
        return float('inf'), {"reason": "visit count out of bounds", "num": num}
    total_dist = 0.0
    total_red = 0.0
    for i in range(num):
        a = selected[i]
        b = selected[(i+1)%num]
        total_dist += dist_matrix[a][b]
        total_red += red_matrix[a][b]
    avg_dist = total_dist / num
    avg_red = total_red / num
    fitness_value = avg_dist - RED_REWARD * avg_red
    if total_red < MIN_RED_TOTAL:
        deficit = (MIN_RED_TOTAL - total_red)
        fitness_value += 1000 * deficit
    return fitness_value, {
        "total_weight": total_dist,
        "total_red": total_red,
        "num": num,
        "raw_cost": total_dist
    }

def init_individual(nodes, idx_map):
    N = len(nodes)
    k = random.randint(MIN_VISIT, MAX_VISIT)
    mask = [0]*N
    fixed_idx = idx_map[FIX_START]
    mask[fixed_idx] = 1
    remaining = [i for i in range(N) if i != fixed_idx]
    choose = k - 1
    chosen = random.sample(remaining, choose)
    for i in chosen:
        mask[i] = 1
    order = nodes.copy()
    random.shuffle(order)
    return (mask, order)

def init_population(nodes, idx_map):
    return [init_individual(nodes, idx_map) for _ in range(POP_SIZE)]

def tournament_selection(pop, fitnesses):
    selected = []
    for _ in range(len(pop)):
        aspirants = random.sample(range(len(pop)), TOURNAMENT_K)
        best = min(aspirants, key=lambda i: fitnesses[i][0])
        selected.append(pop[best])
    return selected

def crossover(parent1, parent2, idx_map):
    mask1, order1 = parent1
    mask2, order2 = parent2
    N = len(mask1)
    child_mask = [mask1[i] if random.random() < 0.5 else mask2[i] for i in range(N)]
    if sum(child_mask) < MIN_VISIT:
        zeros = [i for i in range(N) if child_mask[i] == 0]
        for idx in random.sample(zeros, MIN_VISIT - sum(child_mask)):
            child_mask[idx] = 1
    if sum(child_mask) > MAX_VISIT:
        ones = [i for i in range(N) if child_mask[i] == 1 and i != idx_map[FIX_START]]
        for idx in random.sample(ones, sum(child_mask)-MAX_VISIT):
            child_mask[idx] = 0
    size = len(order1)
    a, b = sorted(random.sample(range(size), 2))
    child_order = [None]*size
    child_order[a:b+1] = order1[a:b+1]
    fill = [x for x in order2 if x not in child_order]
    ptr = 0
    for i in range(size):
        if child_order[i] is None:
            child_order[i] = fill[ptr]; ptr += 1
    return (child_mask, child_order)

def mutate(indiv, idx_map):
    mask, order = indiv
    N = len(mask)
    for i in range(N):
        if i == idx_map[FIX_START]:
            continue
        if random.random() < 0.05:
            mask[i] = 1 - mask[i]
    if sum(mask) < MIN_VISIT:
        zeros = [i for i in range(N) if mask[i]==0 and i != idx_map[FIX_START]]
        for idx in random.sample(zeros, MIN_VISIT - sum(mask)):
            mask[idx]=1
    if sum(mask) > MAX_VISIT:
        ones = [i for i in range(N) if mask[i]==1 and i != idx_map[FIX_START]]
        for idx in random.sample(ones, sum(mask)-MAX_VISIT):
            mask[idx]=0
    if random.random() < MUTATION_PB:
        i, j = random.sample(range(len(order)), 2)
        order[i], order[j] = order[j], order[i]
    return (mask, order)

def rotate_to_start(route, start):
    if start in route:
        idx = route.index(start)
        return route[idx:] + route[:idx]
    return route

def run_ga(edges, red_len, nodes):
    idx_map = {node:i for i,node in enumerate(nodes)}
    dist_matrix, red_matrix = precompute_shortest(nodes, edges, red_len)

    population = init_population(nodes, idx_map)
    best = None
    best_info = None
    history = []

    for gen in range(GENERATIONS):
        fitnesses = [evaluate(ind, dist_matrix, red_matrix) for ind in population]
        fit_vals = [f[0] for f in fitnesses]
        gen_best_idx = min(range(len(fit_vals)), key=lambda i: fit_vals[i])
        if best is None or fit_vals[gen_best_idx] < best_info[0]:
            best = population[gen_best_idx]
            best_info = fitnesses[gen_best_idx]
        history.append(best_info[0])

        selected = tournament_selection(population, fitnesses)

        newpop = []
        sorted_idx = sorted(range(len(fit_vals)), key=lambda i: fit_vals[i])
        for i in range(ELITE_K):
            newpop.append(population[sorted_idx[i]])

        while len(newpop) < POP_SIZE:
            if random.random() < CROSSOVER_PB:
                p1 = random.choice(selected)
                p2 = random.choice(selected)
                child = crossover(p1, p2, idx_map)
            else:
                child = random.choice(selected)
            child = mutate((child[0][:], child[1][:]), idx_map)
            child[0][idx_map[FIX_START]] = 1
            newpop.append(child)
        population = newpop

    return {
        "best_individual": best,
        "best_fitness": best_info[0],
        "details": best_info[1],
        "history": history,
        "idx_map": idx_map,
        "dist_matrix": dist_matrix,
        "red_matrix": red_matrix
    }

def extract_route(individual):
    mask, order = individual
    return [node for keep, node in zip(mask, order) if keep == 1]

def main():
    random.seed(42)
    df = load_csv(CSV_PATH)
    edges, red_len = build_graph(df)
    nodes = sorted(edges.keys())
    result = run_ga(edges, red_len, nodes)

    best = result["best_individual"]
    selected_route = extract_route(최고)
    route = rotate_to_start(selected_route, FIX_START)
    cycle = route + [FIX_START]

    info = result["details"]
    print("=== 최적 부분 순환 경로 ===")
    print("방문 노드 수:", info["num"])
    print("순환 경로:", " -> ".join(cycle))
    print(f"총 다익스트라 기반 거리 합: {info['total_weight']:.1f}")
    print(f"빨간길 길이 합: {info['total_red']:.1f}")
    print(f"방문당 평균 fitness: {result['best_fitness']:.3f}")

if __name__ == "__main__":
    main()
