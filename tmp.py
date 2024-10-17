import math

def euclidean_distance(vector1, vector2):
    if len(vector1) != len(vector2):
        raise ValueError("Vectors must be of same length")
    
    distance = math.sqrt(sum((a - b) ** 2 for a, b in zip(vector1, vector2)))
    return distance

def overlap(vector1, radius1, vector2, radius2):
    distance = euclidean_distance(vector1, vector2)
    print('overlap', distance, radius1, radius2)
    if distance < radius1 + radius2:
        return True
    else:
        return False

# Example usage
vector1 = [5, 5]
radius1 = 3
vector2 = [5, 4]
radius2 = 2

print(euclidean_distance(vector1, vector2))  # Output: 1.0
print(overlap(vector1, radius1, vector2, radius2))  # Output: True