"""Estimate head pose according to the facial landmarks"""
import numpy as np
from pprint import pprint
import cv2




class PoseEstimator:
    """Estimate head pose according to the facial landmarks"""

    # distance to face on Z axis
    AXIS_Z = -500
    POINT_COEFFICIENT = 3
    LIMIT_LINES_COLOR = (60, 80, 60)
    LIMIT_LINES_WIDTH = 2

    def __init__(self, img_size=(480, 640)):
        self.size = img_size

        self.img_height = img_size[0]
        self.img_width = img_size[1]

        # 3D model points.
        self.model_points = np.array([
            (0.0, 0.0, 0.0),             # Nose tip
            (0.0, -330.0, -65.0),        # Chin
            (-225.0, 170.0, -135.0),     # Left eye left corner
            (225.0, 170.0, -135.0),      # Right eye right corne
            (-150.0, -150.0, -125.0),    # Left Mouth corner
            (150.0, -150.0, -125.0)      # Right mouth corner
        ]) / 4.5

        self.model_points_68 = self._get_full_model_points()

        # Camera internals
        self.focal_length = self.size[1]
        self.camera_center = (self.size[1] / 2, self.size[0] / 2)
        self.camera_matrix = np.array(
            [[self.focal_length, 0, self.camera_center[0]],
             [0, self.focal_length, self.camera_center[1]],
             [0, 0, 1]], dtype="double")

        # Assuming no lens distortion
        self.dist_coeefs = np.zeros((4, 1))

        # Rotation vector and translation vector
        self.r_vec = np.array([[0.01891013], [0.08560084], [-3.14392813]])
        self.t_vec = np.array(
            [[-14.97821226], [-10.62040383], [-2053.03596872]])
        # self.r_vec = None
        # self.t_vec = None

        self._3d_box = self.draw_3d_box_as_annotation_of_pose()

    def _get_full_model_points(self, filename='assets/model.txt'):
        """Get all 68 3D model points from file"""
        raw_value = []
        with open(filename) as file:
            for line in file:
                raw_value.append(line)
        model_points = np.array(raw_value, dtype=np.float32)
        model_points = np.reshape(model_points, (3, -1)).T
        # model_points *= 4
        model_points[:, -1] *= -1

        return model_points

    def solve_pose(self, image_points):
        """
        Solve pose from image points
        Return (rotation_vector, translation_vector) as pose.
        """
        (_, rotation_vector, translation_vector) = cv2.solvePnP(
            self.model_points, image_points, self.camera_matrix, self.dist_coeefs)

        (success, rotation_vector, translation_vector) = cv2.solvePnP(
            self.model_points,
            image_points,
            self.camera_matrix,
            self.dist_coeefs,
            rvec=self.r_vec,
            tvec=self.t_vec,
            useExtrinsicGuess=True)
        return (rotation_vector, translation_vector)

    def solve_pose_by_68_points(self, image_points):
        """
        Solve pose from all the 68 image points
        Return (rotation_vector, translation_vector) as pose.
        """

        if self.r_vec is None:
            (_, rotation_vector, translation_vector) = cv2.solvePnP(
                self.model_points_68, image_points, self.camera_matrix, self.dist_coeefs)
            self.r_vec = rotation_vector
            self.t_vec = translation_vector

        (_, rotation_vector, translation_vector) = cv2.solvePnP(
            self.model_points_68,
            image_points,
            self.camera_matrix,
            self.dist_coeefs,
            rvec=self.r_vec,
            tvec=self.t_vec,
            useExtrinsicGuess=True)

        return (rotation_vector, translation_vector)

    def draw_annotation_box(self,
                            image,
                            rotation_vector,
                            translation_vector,
                            color=(255, 255, 255),
                            line_width=LIMIT_LINES_WIDTH):

        point_2d = self.draw_2d_poins_based_on_3d_box_of_pose(rotation_vector, translation_vector)

        # Draw all the lines
        cv2.polylines(image, [point_2d], True, color, line_width, cv2.LINE_AA)
        cv2.line(image, tuple(point_2d[1]), tuple(
            point_2d[6]), color, line_width, cv2.LINE_AA)
        cv2.line(image, tuple(point_2d[2]), tuple(
            point_2d[7]), color, line_width, cv2.LINE_AA)
        cv2.line(image, tuple(point_2d[3]), tuple(
            point_2d[8]), color, line_width, cv2.LINE_AA)

    def draw_2d_poins_based_on_3d_box_of_pose(self, rotation_vector, translation_vector):
        # Map to 2d image points
        (point_2d, _) = cv2.projectPoints(self._3d_box,
                                          rotation_vector,
                                          translation_vector,
                                          self.camera_matrix,
                                          self.dist_coeefs)

        point_2d = np.int32(point_2d.reshape(-1, 2))

        return point_2d

    def draw_3d_box_as_annotation_of_pose(self):
        """Draw a 3D box as annotation of pose"""
        point_3d = []
        box_sizes_depth = [
            [75, 0],
            [100, 100],
        ]
        for size, depth in box_sizes_depth:
            self.append_3d_square_points(point_3d, size, depth)
        point_3d = np.array(point_3d, dtype=np.float).reshape(-1, 3)
        return point_3d

    def append_3d_square_points(self, point_3d, size, depth):
        point_3d.append((-size, -size, depth))
        point_3d.append((-size, size, depth))
        point_3d.append((size, size, depth))
        point_3d.append((size, -size, depth))
        point_3d.append((-size, -size, depth))

    def create_limit_points_bottom(self, size, depth=0):
        line_map = [(size, -size, depth), (-size, -size, depth)]
        line_map = np.array(line_map, dtype=np.float).reshape(-1, 3)
        return line_map

    def create_limit_points_top(self, size, depth=0):
        line_map = [(size, size, depth), (-size, size, depth)]
        line_map = np.array(line_map, dtype=np.float).reshape(-1, 3)
        return line_map

    def create_limit_points_right(self, size, depth=0):
        line_map = [(size, -size, depth), (size, size, depth)]
        line_map = np.array(line_map, dtype=np.float).reshape(-1, 3)
        pprint(line_map)
        # exit(1)
        return line_map

    def create_limit_points_left(self, size, depth=0):
        line_map = [(size, -size, depth), (size, size, depth)]
        line_map = np.array(line_map, dtype=np.float).reshape(-1, 3)
        return line_map

    def draw_limit_line_top(self, image, top, color=LIMIT_LINES_COLOR, line_width=LIMIT_LINES_WIDTH):
        # limit_points = self.create_limit_points_top(abs(top * self.POINT_COEFFICIENT))
        # translation = (0, top, self.AXIS_Z)
        # points_2d = self.transform_limit_points_to_image_coords(limit_points, translation)
        points_2d = [
            [0, 200],
            [self.img_width, 200]
        ]
        points_2d = np.array(points_2d, dtype=np.int32).reshape(2, 2)

        cv2.polylines(image, [points_2d], True, color, line_width, cv2.LINE_AA)

    def draw_limit_line_bottom(self, image, bottom, color=LIMIT_LINES_COLOR, line_width=LIMIT_LINES_WIDTH):
        # translation = (0, bottom, self.AXIS_Z)
        # limit_points = self.create_limit_points_bottom(abs(bottom * self.POINT_COEFFICIENT))
        #
        # points_2d = self.transform_limit_points_to_image_coords(limit_points, translation)
        # points_2d[0][0] = 0
        # points_2d[1][0] = self.img_width
        # pprint(points_2d)
        points_2d = [
            [0, 600],
            [self.img_width, 600]
        ]
        points_2d = np.array(points_2d, dtype=np.int32).reshape(2, 2)
        cv2.polylines(image, [points_2d], True, color, line_width, cv2.LINE_AA)

    def draw_limit_line_right(self, image, right, color=LIMIT_LINES_COLOR, line_width=LIMIT_LINES_WIDTH):
        # translation = (right, 0, self.AXIS_Z)
        # limit_points = self.create_limit_points_right(abs(right * self.POINT_COEFFICIENT))
        #
        # points_2d = self.transform_limit_points_to_image_coords(limit_points, translation)
        # points_2d[0][1] = 0
        # points_2d[1][1] = self.img_height
        # pprint(points_2d)
        points_2d = [
            [800, 0],
            [800, self.img_height]
        ]
        points_2d = np.array(points_2d, dtype=np.int32).reshape(2, 2)

        cv2.polylines(image, [points_2d], True, color, line_width, cv2.LINE_AA)

    def draw_limit_line_left(self, image, left, color=LIMIT_LINES_COLOR, line_width=LIMIT_LINES_WIDTH):
        # translation = (left, 0, self.AXIS_Z)
        # limit_points = self.create_limit_points_left(abs(left * self.POINT_COEFFICIENT))
        #
        # points_2d = self.transform_limit_points_to_image_coords(limit_points, translation)
        # points_2d[0][1] = 0
        # points_2d[1][1] = self.img_height
        # # pprint(points_2d)

        points_2d = [
            [400, 0],
            [400, self.img_height]
        ]
        points_2d = np.array(points_2d, dtype=np.int32).reshape(2, 2)
        cv2.polylines(image, [points_2d], True, color, line_width, cv2.LINE_AA)


    def transform_limit_points_to_image_coords(self, object_points, translation):
        zero_rotation = []
        zero_rotation.append((0, 0, 0))
        zero_rotation = np.array(zero_rotation, dtype=np.float32).reshape(1, 3)

        limit_translation = []
        limit_translation.append(translation)
        limit_translation = np.array(limit_translation, dtype=np.float32).reshape(1, 3)

        (point_2d, _) = cv2.projectPoints(object_points,
                                          zero_rotation,
                                          limit_translation,
                                          self.camera_matrix,
                                          self.dist_coeefs)

        point_2d = np.int32(point_2d.reshape(-1, 2))
        return point_2d

    def get_pose_marks(self, marks):
        """Get marks ready for pose estimation from 68 marks"""
        pose_marks = []
        pose_marks.append(marks[30])    # Nose tip
        pose_marks.append(marks[8])     # Chin
        pose_marks.append(marks[36])    # Left eye left corner
        pose_marks.append(marks[45])    # Right eye right corner
        pose_marks.append(marks[48])    # Left Mouth corner
        pose_marks.append(marks[54])    # Right mouth corner
        return pose_marks
