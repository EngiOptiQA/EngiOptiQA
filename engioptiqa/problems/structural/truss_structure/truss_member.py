class TrussMember:
    def __init__(self, node_id_0, node_id_1, coords_0, coords_1, A=None, E=None, member_id=None):
        self.node_id_0 = node_id_0
        self.node_id_1 = node_id_1
        self.coords = [coords_0, coords_1]
        self.A = A
        self.A_initial = A
        self.E = E
        self.member_id = member_id
        self.length = self.compute_length()
        self.direction_cosines_0 = self.compute_direction_cosines(0)
        self.direction_cosines_1 = self.compute_direction_cosines(1)
        self.exists = True

    def get_coords(self, local_node_id):
        if local_node_id not in [0, 1]:
            raise ValueError("local_node_id must be either 0 or 1")
        return self.coords[local_node_id]

    def compute_length(self):
        x0, y0 = self.get_coords(0)
        x1, y1 = self.get_coords(1)
        return ((x1 - x0)**2 + (y1 - y0)**2)**0.5

    def compute_direction_cosines(self, local_node_id):
        if local_node_id not in (0, 1):
            raise ValueError("local_node_id must be either 0 or 1")

        x0, y0 = self.get_coords(0)
        x1, y1 = self.get_coords(1)

        sign = 1 if local_node_id == 0 else -1
        L = self.length

        l = sign * (x1 - x0) / L
        m = sign * (y1 - y0) / L

        return (l, m)

