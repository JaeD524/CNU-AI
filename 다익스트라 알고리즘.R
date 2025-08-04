# 📦 라이브러리 불러오기
library(readxl)
library(dplyr)


edges <- read_excel("/Users/bagjaeyong/Desktop/대학교/2025-여름/충남대 생성형 AI/노선 가중치.xlsx") %>%
  select(출발점, 도착점, 가중치, 빨간길 = `빨간길(m)`) %>%
  filter(!is.na(출발점), !is.na(도착점), !is.na(가중치)) %>%
  mutate(총비용 = pmax(0, 가중치 - 2 * 빨간길))  # 빨간길 있을수록 비용 ↓


nodes <- unique(c(edges$출발점, edges$도착점))
n <- length(nodes)
node_ids <- setNames(1:n, nodes)
id_to_node <- setNames(nodes, 1:n)


adj_list <- vector("list", length = n)
names(adj_list) <- nodes

for (i in 1:nrow(edges)) {
  from <- edges$출발점[i]
  to <- edges$도착점[i]
  cost <- edges$총비용[i]
  
  adj_list[[from]] <- append(adj_list[[from]], list(list(to = to, cost = cost)))
  adj_list[[to]] <- append(adj_list[[to]], list(list(to = from, cost = cost)))
}


dijkstra <- function(start_node) {
  dist <- rep(Inf, n)
  names(dist) <- nodes
  dist[start_node] <- 0
  
  visited <- rep(FALSE, n)
  names(visited) <- nodes
  
  for (i in 1:n) {
    current <- names(which.min(ifelse(visited, Inf, dist)))
    if (is.infinite(dist[current])) break
    visited[current] <- TRUE
    
    for (neighbor in adj_list[[current]]) {
      to <- neighbor$to
      cost <- neighbor$cost
      if (!visited[to] && dist[current] + cost < dist[to]) {
        dist[to] <- dist[current] + cost
      }
    }
  }
  
  return(dist)
}


start_node <- "기숙사"  
최소거리 <- dijkstra(start_node)

# 6️⃣ 결과 출력
cat("🚍", start_node, "에서 각 노드까지 최소 비용 경로 거리:\n")
print(최소거리)


total_costs <- sapply(nodes, function(start) {
  dist <- dijkstra(start)
  sum(dist[!is.infinite(dist) & names(dist) != start])  
})


total_costs <- round(total_costs, 1)
print(total_costs)


best_start <- names(which.min(total_costs))
cat("최적 출발 정류장:", best_start, " (총 거리:", total_costs[best_start], ")\n")

