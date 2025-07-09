import math

class Tracker:
    def __init__(self, max_disappeared=10, distance_threshold=60):
        self.next_id = 0
        self.objects = {}  # id -> (center_x, center_y)
        self.disappeared = {}  # id -> frame count
        self.max_disappeared = max_disappeared
        self.distance_threshold = distance_threshold

    def _calculate_distance(self, point1, point2):
        return math.hypot(point1[0] - point2[0], point1[1] - point2[1])

    def update(self, rects):
        objects_bbs_ids = []

        # No detections
        if len(rects) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    del self.objects[object_id]
                    del self.disappeared[object_id]
            return []

        input_centers = []
        for rect in rects:
            x1, y1, x2, y2 = rect
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            input_centers.append(((x1, y1, x2, y2), (cx, cy)))

        used_ids = set()
        matched_ids = set()

        # Match new detections to existing objects
        for rect, center in input_centers:
            best_id = None
            best_distance = self.distance_threshold

            for object_id, object_center in self.objects.items():
                if object_id in used_ids:
                    continue
                distance = self._calculate_distance(center, object_center)
                if distance < best_distance:
                    best_distance = distance
                    best_id = object_id

            if best_id is not None:
                self.objects[best_id] = center
                self.disappeared[best_id] = 0
                used_ids.add(best_id)
                matched_ids.add(best_id)
                x1, y1, x2, y2 = rect
                objects_bbs_ids.append([x1, y1, x2, y2, best_id])
            else:
                # New object
                self.objects[self.next_id] = center
                self.disappeared[self.next_id] = 0
                x1, y1, x2, y2 = rect
                objects_bbs_ids.append([x1, y1, x2, y2, self.next_id])
                self.next_id += 1

        # Update disappearance counter for unmatched IDs
        for object_id in list(self.objects.keys()):
            if object_id not in matched_ids:
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    del self.objects[object_id]
                    del self.disappeared[object_id]

        return objects_bbs_ids
