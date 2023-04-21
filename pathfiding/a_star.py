import numpy as np
import heapq


def astar_path(map, start, end):

    came_from = {}
    came_from[start] = None

    # Define the heuristic function as the Manhattan distance
    def heuristic(a, b):
        return np.abs(a[0] - b[0]) + np.abs(a[1] - b[1])

    # Define the cost function for moving from one cell to the next
    def cost(current, next):
        return map[next[0], next[1]]

    # Initialize the open and closed sets
    open_set = []
    closed_set = set()

    # Add the start node to the open set with a cost of 0
    heapq.heappush(open_set, (0, start))

    # Initialize the g score (cost from start to current) and f score (g score + heuristic) for the start node
    g_score = {start: 0}
    f_score = {start: heuristic(start, end)}

    # Start the A* search
    while open_set:
        # Get the node with the lowest f score from the open set
        current_cost, current_node = heapq.heappop(open_set)

        # Check if we've reached the target node
        if current_node == end:
            path = [current_node]
            while current_node in came_from:
                current_node = came_from[current_node]
                path.append(current_node)
            path.reverse()
            return path

        # Add the current node to the closed set
        closed_set.add(current_node)

        # Check all the neighboring nodes
        for x, y in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            next_node = current_node[0] + x, current_node[1] + y

            # Skip the node if it's outside the map or if it's an obstacle
            if not (0 <= next_node[0] < map.shape[0] and 0 <= next_node[1] < map.shape[1] and map[next_node[0], next_node[1]] > 0):
                continue

            # Calculate the tentative g score (cost from start to next) for the next node
            tentative_g_score = g_score[current_node] + cost(current_node, next_node)

            # Skip this node if we've already visited it with a lower g score
            if next_node in closed_set and tentative_g_score >= g_score.get(next_node, float('inf')):
                continue

            # Add this node to the open set if we've never visited it or if we've found a shorter path to it
            if tentative_g_score < g_score.get(next_node, float('inf')):
                came_from[next_node] = current_node
                g_score[next_node] = tentative_g_score
                f_score[next_node] = tentative_g_score + heuristic(next_node, end)
                heapq.heappush(open_set, (f_score[next_node], next_node))

    # If we've exhausted the open set without finding the target, there's no path
    return None
