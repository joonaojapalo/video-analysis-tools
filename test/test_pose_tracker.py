import unittest
import pprint

import pose_tracker


def keypoints(x0, y0, x1, y1):
    return [x0, y0, 1, x1, y0, 1, x1, y1, 1, x0, y1, 1]


def obj(idx, points):
    x0, y0, x1, y1 = points
    return {
        "keypoints": keypoints(x0, y0, x1, y1),
        "idx": idx,
    }


def frame(frame_num, data):
    return {
        "image_id": f"{frame_num}.jpg",
        "objs": [obj(idx, xy) for idx, xy in data]
    }


def b(cx, cy, w=10, h=10):
    return [cx-w//2, cy-h//2, cx+w//2, cy+h//2]


def o(idx, cx, cy):
    return (idx, b(cx, cy))


class TestPoseTracker (unittest.TestCase):
    def test_normal(self):
        sequence = [
            frame(0, [o(1, 5, 5)]),
            frame(1, [o(1, 10, 10)]),
            frame(2, [o(1, 15, 15)]),
        ]
        evs, duplicates = pose_tracker.get_pose_idx_events(sequence)
        self.assertEqual(len(evs), 0)
        self.assertEqual(len(duplicates), 0)

    def test_normal2(self):
        sequence = [
            frame(0, [o(1, 5, 5), o(2, 20, 20)]),
            frame(1, [o(1, 10, 10), o(2, 30, 20)]),
            frame(3, [o(1, 15, 15), o(2, 40, 20)]),
        ]
        evs, duplicates = pose_tracker.get_pose_idx_events(sequence)
        self.assertEqual(len(evs), 0)
        self.assertEqual(len(duplicates), 0)

    def test_idx_switch01(self):
        sequence = [
            frame(0, [o(1, 5, 100)]),
            frame(1, [o(1, 6, 100)]),
            frame(2, [o(2, 7, 100)]),
        ]
        evs, duplicates = pose_tracker.get_pose_idx_events(sequence)
        self.assertEqual(len(evs), 1)
        self.assertEqual(len(duplicates), 0)

    def test_idx_switch02(self):
        sequence = [
            frame(0, [o(1, 5, 100)]),
            frame(1, [o(1, 6, 100)]),
            frame(2, [o(2, 7, 100)]),
            frame(3, [o(1, 8, 100)]),
        ]
        evs, duplicates = pose_tracker.get_pose_idx_events(sequence)
        self.assertEqual(len(evs), 2, evs)
        self.assertEqual(len(duplicates), 0)

    def test_idx_switch03(self):
        sequence = [
            frame(0, [o(1, 5, 100), o(3, 5, 200)]),
            frame(1, [o(1, 6, 100), o(3, 6, 200)]),
            frame(2, [o(2, 7, 100), o(3, 7, 200)]),
            frame(3, [o(2, 8, 100), o(3, 8, 200)]),
            frame(4, [o(2, 9, 100), o(3, 9, 200)]),
            frame(5, [o(1, 10, 100), o(3, 10, 200)]),
            frame(6, [o(1, 11, 100), o(3, 11, 200)]),
            frame(7, [o(1, 12, 100), o(3, 12, 200)]),
        ]
        evs, duplicates = pose_tracker.get_pose_idx_events(sequence)
        self.assertEqual(len(evs), 2)
        self.assertEqual(len(duplicates), 0)

    def test_idx_switch04(self):
        sequence = [
            frame(0, [o(1, 5, 100), o(3, 5, 200)]),
            frame(1, [o(1, 6, 100), o(3, 6, 200)]),
            frame(2, [o(2, 7, 100), o(3, 7, 200)]),
            frame(3, [o(2, 8, 100), o(3, 8, 200)]),
            frame(4, [o(2, 9, 100), o(4, 9, 200)]),
            frame(5, [o(1, 10, 100), o(4, 10, 200)]),
            frame(6, [o(1, 11, 100), o(4, 11, 200)]),
            frame(7, [o(1, 12, 100), o(3, 12, 200)]),
        ]
        evs, duplicates = pose_tracker.get_pose_idx_events(sequence)
        self.assertEqual(len(evs), 4)
        self.assertEqual(len(duplicates), 0)

    def test_idx_duplicate01(self):
        sequence = [
            frame(0, [o(1, 5, 100)]),
            frame(1, [o(1, 6, 100)]),
            frame(2, [o(1, 7, 100), o(2, 7, 100)]),
            frame(3, [o(1, 8, 100)]),
        ]
        evs, duplicates = pose_tracker.get_pose_idx_events(sequence)
        self.assertEqual(len(evs), 0, "Should remove duplicate")
        self.assertEqual(len(duplicates), 1)

    def test_idx_duplicate02(self):
        sequence = [
            frame(0, [o(1, 5, 100)]),
            frame(1, [o(1, 6, 100)]),
            frame(2, [o(1, 7, 100), o(2, 7, 100)]),
            frame(3, [o(2, 8, 100)]),
        ]
        evs, duplicates = pose_tracker.get_pose_idx_events(sequence)
        self.assertEqual(len(evs), 1,
                         "Should remove duplicate and track index change")
        self.assertEqual(len(duplicates), 1)

    def test_idx_duplicate03(self):
        sequence = [
            #            frame(0, [o(1, 5, 100), o(2, 5, 100)]),
            frame(0, [o(1, 5, 100)]),
            frame(1, [o(1, 5, 100), o(2, 5, 100)]),
            frame(2, [o(1, 6, 100), o(2, 6, 100)]),
            frame(3, [o(1, 7, 100), o(2, 7, 100)]),
            frame(4, [o(1, 8, 100)]),
        ]
        evs, duplicates = pose_tracker.get_pose_idx_events(sequence)
        self.assertEqual(len(evs), 0, "Should remove sequent duplicates")
        self.assertEqual(len(duplicates), 3)

    def xtest_idx_duplicates_first_frame(self):
        sequence = [
            frame(0, [o(1, 5, 100), o(2, 5, 100), o(3, 5, 200)]),
            frame(1, [o(1, 6, 100), o(2, 6, 100), o(4, 6, 200)]),
            frame(2, [o(1, 7, 100), o(2, 7, 100), o(3, 7, 200)]),
            frame(3, [o(1, 8, 100)]),
        ]
        evs, duplicates = pose_tracker.get_pose_idx_events(sequence)
        self.assertEqual(len(evs), 2, "Should remove sequent duplicates")
        self.assertEqual(len(duplicates), 3)

    def test_remap01(self):
        sequence = [
            frame(1, [o(1, 100, 100)]),
            frame(2, [o(1, 101, 100), o(5, 101, 101)]),
        ]

        # changes
        duplicates = [['2.jpg', 5]]
        pose_tracker.remap_idx_inplace(sequence, [], duplicates)
        #pprint.pprint(sequence)
        expected_sequence = [
            frame(1, [o(1, 100, 100)]),
            frame(2, [o(1, 101, 100)]),
        ]
        self.assertEqual(sequence, expected_sequence)

    def test_remap02(self):
        sequence = [
            frame(1, [o(1, 100, 100)]),
            frame(2, [o(1, 101, 100), o(5, 101, 100.5)]),
            frame(3, [o(1, 102, 100), o(5, 102, 100.5)]),
            frame(4, [o(5, 103, 100.5)]),
        ]

        # remap
        pose_tracker.remap_idx_inplace(sequence, [["4.jpg", 5, 1]], [
                                       ["2.jpg", 5], ["3.jpg", 5]])

        expected_sequence = [
            frame(1, [o(10001, 100, 100)]),
            frame(2, [o(10001, 101, 100)]),
            frame(3, [o(10001, 102, 100)]),
            frame(4, [o(10001, 103, 100.5)]),
        ]
        self.assertEqual(sequence, expected_sequence)

    def test_remap03(self):
        sequence = [
            frame(1, [o(1, 100, 100)]),
            frame(2, [o(1, 101, 100), o(5, 101, 100.5)]),
            frame(3, [o(1, 102, 100), o(5, 102, 100.5)]),
            frame(4, [o(5, 103, 100.5)]),
            frame(5, [o(1, 104, 100), o(5, 104, 100.5)]),
            frame(6, [o(1, 105, 100), o(5, 105, 100.5)]),
            frame(7, [o(1, 106, 100)]),
            frame(8, [o(1, 107, 100), o(5, 107, 100.5)]),
            frame(9, [o(1, 108, 100), o(5, 108, 100.5)]),
        ]
        changes, duplicates = pose_tracker.get_pose_idx_events(sequence)

        self.assertEqual(len(changes), 2, "Should detect 2 changes")
        self.assertEqual(len(duplicates), 6, "Should detect duplicates")

        # changes
        self.assertTrue(["4.jpg", 5, 1] in changes)
        self.assertTrue(["7.jpg", 1, 5] in changes)

        # dupes
        self.assertTrue(["2.jpg", 5] in duplicates)
        self.assertTrue(["3.jpg", 5] in duplicates)
        self.assertTrue(["5.jpg", 1] in duplicates)
        self.assertTrue(["6.jpg", 1] in duplicates)
        self.assertTrue(["8.jpg", 5] in duplicates)
        self.assertTrue(["9.jpg", 5] in duplicates)

        # remap test
        pose_tracker.remap_idx_inplace(sequence, changes, duplicates)

        expected_sequence = [
            frame(1, [o(10001, 100, 100)]),
            frame(2, [o(10001, 101, 100)]),
            frame(3, [o(10001, 102, 100)]),
            frame(4, [o(10001, 103, 100.5)]),
            frame(5, [o(10001, 104, 100.5)]),
            frame(6, [o(10001, 105, 100.5)]),
            frame(7, [o(10001, 106, 100)]),
            frame(8, [o(10001, 107, 100)]),
            frame(9, [o(10001, 108, 100)]),
        ]
        self.assertEqual(sequence, expected_sequence)

    def test_multiple_duplicates(self):
        sequence = [
            frame(1, [o(1, 100, 100)]),
            frame(2, [o(1, 101, 100), o(5, 101, 100.25), o(6, 101, 100.5)]),
        ]
        changes, duplicates = pose_tracker.get_pose_idx_events(sequence)
        self.assertEqual(len(duplicates), 2)

        pose_tracker.remap_idx_inplace(sequence, changes, duplicates)

        expected_sequence = [
            frame(1, [o(1, 100, 100)]),
            frame(2, [o(1, 101, 100)])
        ]
        self.assertEqual(sequence, expected_sequence)

    def test_gap(self):
        sequence = [
            frame(0, [o(1, 5, 100)]),
            frame(1, [o(1, 6, 100)]),
            frame(2, []),
            frame(3, [o(2, 7, 100)]),
            frame(4, [o(2, 8, 100)]),
        ]
        changes, duplicates = pose_tracker.get_pose_idx_events(sequence)
        self.assertEqual(len(duplicates), 0)
        self.assertEqual(len(changes), 1)
        self.assertEqual(changes[0], ['3.jpg', 2, 1])
