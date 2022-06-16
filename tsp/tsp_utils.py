import numpy as np
import scipy.spatial
import cv2

class TSP_2opt():
    def __init__(self, points):
        self.dist_mat = scipy.spatial.distance_matrix(points, points)
    
    def evaluate(self, route):
        total_cost = 0
        for i in range(len(route)-1):
            total_cost += self.dist_mat[route[i],route[i+1]]
        return total_cost

    def solve_2opt(self, route):
        assert route[0] == route[-1], 'Tour is not a cycle'

        best = route
        improved = True
        steps = 0
        while improved:
            improved = False
            for i in range(1, len(route)-2):
                for j in range(i+1, len(route)):
                    if j-i == 1: continue # changes nothing, skip then
                    new_route = route[:]
                    new_route[i:j] = route[j-1:i-1:-1] # this is the 2woptSwap
                    if self.evaluate(new_route) < self.evaluate(best):
                        best = new_route
                        steps += 1
                        improved = True
            route = best
        return best, steps
    
    def seed_solver(self, routes):
        best = self.solve_2opt(routes[0])
        for i in range(1, len(routes)):
            result = self.solve_2opt(routes[i])
            if self.evaluate(result) < self.evaluate(best):
                best = result
        return best

def rasterize_tsp(points, tour, img_size, line_color, line_thickness, point_color, point_radius):
    # Rasterize lines
    img = np.zeros((img_size, img_size))
    for i in range(len(tour)-1):
        from_idx = tour[i]
        to_idx = tour[i+1]

        cv2.line(img, 
                 tuple(((img_size-1)*points[from_idx,::-1]).astype(int)), 
                 tuple(((img_size-1)*points[to_idx,::-1]).astype(int)), 
                 color=line_color, thickness=line_thickness)

    # Rasterize points
    for i in range(points.shape[0]):
        cv2.circle(img, tuple(((img_size-1)*points[i,::-1]).astype(int)), 
                   radius=point_radius, color=point_color, thickness=-1)

    return img
