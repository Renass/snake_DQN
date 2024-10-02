from collections import deque

def check_head_to_tail_accessibility(snake):
    """
    This function checks if the snake's head can reach the snake's tail 
    without hitting its own body or other obstacles.
    """
    visited = [[False for _ in range(30)] for _ in range(30)]  # 30x30 grid
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Possible directions (up, down, left, right)
    
    head = snake.body[0]  # Snake's head (starting point)
    tail = snake.body[-1]  # Snake's tail (goal point)
    
    queue = deque([head])  # Initialize BFS queue with the head position
    visited[head[0]][head[1]] = True  # Mark the head position as visited
    
    # Mark snake's body (excluding the head and tail) as visited to avoid crossing over itself
    for x, y in snake.body[1:-1]:  # Exclude the head and tail from being marked as visited
        visited[x][y] = True

    # BFS loop to find if a path exists from head to tail
    while queue:
        x, y = queue.popleft()

        # If we reach the tail, return True
        if (x, y) == tail:
            return True
        
        # Explore the neighboring cells
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < 30 and 0 <= ny < 30 and not visited[nx][ny]:
                visited[nx][ny] = True
                queue.append((nx, ny))
    
    # If we finish BFS without reaching the tail, return False
    return False








