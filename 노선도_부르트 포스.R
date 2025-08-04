library(gtools) 
library(dplyr)
library(readxl)


edges <- read_excel("/Users/bagjaeyong/Desktop/대학교/2025-여름/충남대 생성형 AI/노선 가중치.xlsx") %>%
  select(출발점, 도착점, 가중치, 빨간길 = `빨간길(m)`) %>%
  filter(!is.na(출발점), !is.na(도착점), !is.na(가중치))


nodes <- unique(c(edges$출발점, edges$도착점))
start_node <- "도서관"
other_nodes <- setdiff(nodes, start_node)


create_distance_matrix <- function(edges) {
  mat <- matrix(Inf, length(nodes), length(nodes))
  rownames(mat) <- nodes
  colnames(mat) <- nodes
  for (i in 1:nrow(edges)) {
    from <- edges$출발점[i]
    to <- edges$도착점[i]
    weight <- edges$가중치[i] + edges$빨간길[i] * 5
    mat[from, to] <- weight
    mat[to, from] <- weight
  }
  return(mat)
}
distance_matrix <- create_distance_matrix(edges)


brute_force_route <- function(start_node, other_nodes, n_stop, distance_matrix) {
  min_cost <- Inf
  best_path <- NULL
  
  combinations <- combn(other_nodes, n_stop - 1, simplify = FALSE)
  total <- length(combinations)
  cat("🔍총 조합 수:", total, "\n")
  
  for (combo in combinations) {
    perms <- permutations(n = length(combo), r = length(combo), v = combo)
    
    for (i in 1:nrow(perms)) {
      route <- c(start_node, perms[i, ])
      cost <- 0
      valid <- TRUE
      
      for (j in 1:(length(route) - 1)) {
        from <- route[j]
        to <- route[j + 1]
        d <- distance_matrix[from, to]
        if (is.na(d) || is.infinite(d)) {
          valid <- FALSE
          break
        }
        cost <- cost + d
      }
      
      if (valid && cost < min_cost) {
        min_cost <- cost
        best_path <- route
      }
    }
  }
  
  return(list(path = best_path, cost = min_cost))
}


set.seed(42)
n_stop <- 8  
result <- brute_force_route(start_node, other_nodes, n_stop, distance_matrix)


cat(" 브루트포스 최적 경로:\n", paste(result$path, collapse = " → "), "\n총 거리:", round(result$cost, 1), "\n")
