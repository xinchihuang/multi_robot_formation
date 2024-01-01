def find_connected_components_with_count(matrix):
    def dfs(r, c, component_number):
        if r < 0 or r >= rows or c < 0 or c >= cols or matrix[r][c] != 1:
            return 0

        matrix[r][c] = component_number
        count = 1  # Count the current cell

        for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            count += dfs(r + dr, c + dc, component_number)

        return count

    rows = len(matrix)
    cols = len(matrix[0])
    component_number = 2  # Start component numbering from 2
    component_counts = {}  # Dictionary to store counts for each component

    for r in range(rows):
        for c in range(cols):
            if matrix[r][c] == 1:
                count = dfs(r, c, component_number)
                component_counts[component_number] = count
                component_number += 1

    return matrix, component_counts

# Example usage:
binary_matrix = [
    [1, 1, 0, 0, 0],
    [1, 1, 0, 0, 1],
    [0, 0, 0, 1, 1],
    [0, 0, 0, 0, 0],
]

connected_components, component_counts = find_connected_components_with_count(binary_matrix)

for row in connected_components:
    print(row)

for component_number, count in component_counts.items():
    print(f"Component {component_number}: {count} '1's")
