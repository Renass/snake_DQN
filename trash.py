from collections import deque

def check_field_accessibility(snake):
    # Initialize a visited matrix for the 30x30 grid
    visited = [[False for _ in range(30)] for _ in range(30)]
    accessible_cells = 0
    total_free_cells = 900 - len(snake.body)  # Total free cells excluding snake's body
    required_accessible_cells = total_free_cells * 0.8  # 80% of free cells
    
    # Directions for moving up, down, left, and right
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    # The snake's head is the starting point for BFS
    head_x, head_y = snake.body[0]
    queue = deque([(head_x, head_y)])  # Queue starts from the snake's head
    visited[head_x][head_y] = True
    
    # Mark snake's body as visited to avoid counting them as accessible
    for x, y in snake.body:
        visited[x][y] = True
    
    # BFS to explore accessible cells
    while queue:
        x, y = queue.popleft()
        accessible_cells += 1
        
        # Early stopping if we have already found enough accessible cells
        if accessible_cells >= required_accessible_cells:
            return True
        
        # Explore the neighboring cells
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < 30 and 0 <= ny < 30 and not visited[nx][ny]:
                visited[nx][ny] = True  # Mark as visited
                queue.append((nx, ny))
    
    # Return whether the accessible cells are >= required accessible cells
    return accessible_cells >= required_accessible_cells








