import numpy as np
import matplotlib.pyplot as plt

# Function to paint the circle on the grid based on real-space coordinates
def paint_circle(grid, center, radius, grid_size, grid_resolution):
    # Convert real-space coordinates to grid coordinates
    cx, cy = get_cell_index(center[0], center[1], grid_size, grid_resolution)
    
    # Convert the real-space radius to grid radius
    grid_radius = int((radius / grid_size) * grid_resolution)

    # Paint the circle
    for x in range(cx - grid_radius, cx + grid_radius + 1):
        for y in range(cy - grid_radius, cy + grid_radius + 1):
            if (x - cx)**2 + (y - cy)**2 <= grid_radius**2:
                if 0 <= x < grid.shape[0] and 0 <= y < grid.shape[1]:
                    grid[x, y] = 1

# Function to determine which cell a given (x, y) real-space value belongs to
def get_cell_index(x, y, grid_size, grid_resolution):
    cell_x = int((x / grid_size) * grid_resolution)
    cell_y = int((y / grid_size) * grid_resolution)
    
    # Ensure the indices are within bounds
    cell_x = min(max(cell_x, 0), grid_resolution - 1)
    cell_y = min(max(cell_y, 0), grid_resolution - 1)
    
    return cell_x, cell_y

def main():
    # Parameters
    grid_resolution = 1000  # Grid resolution (1000x1000)
    grid_size = 1000  # Real-world space, for instance, a 100x100 unit space
    
    # Create a grid
    grid = np.zeros((grid_resolution, grid_resolution), dtype=int)

    # Define circle parameters in real-world coordinates
    circle_center = (500, 500)  # Circle center in real-world coordinates
    circle_radius = 10  # Circle radius in real-world units

    # Paint the circle on the grid based on real-space coordinates and radius
    paint_circle(grid, circle_center, circle_radius, grid_size, grid_resolution)

    # Plotting the grid
    plt.imshow(grid, cmap='Greys', origin='lower')
    plt.title(f'Circle with center {circle_center} and radius {circle_radius}')
    plt.show()

    # Example of mapping real space to cell index
    real_x, real_y = 45, 70  # Some point in real-world space
    cell_x, cell_y = get_cell_index(real_x, real_y, grid_size, grid_resolution)
    print(f"Real-space point ({real_x}, {real_y}) maps to grid cell ({cell_x}, {cell_y})")

if __name__ == "__main__":
    main()
