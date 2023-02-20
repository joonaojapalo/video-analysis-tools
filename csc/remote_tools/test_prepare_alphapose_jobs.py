import unittest

from prepare_alphapose_jobs import compute_job_item_allocation

class TestJobAllocation (unittest.TestCase):
    def test_compute_job_item_allocation_64(self):
        alloc = compute_job_item_allocation(64, 32)
        self.assertEqual(alloc["ARRAY_ITEMS"], 32)
        self.assertEqual(alloc["JOBS_PER_ITEM"], 2)

    def test_compute_job_item_allocation_68(self):
        alloc = compute_job_item_allocation(68)
        self.assertGreaterEqual(alloc["ARRAY_ITEMS"] * alloc["JOBS_PER_ITEM"], 68)

    def test_compute_job_item_allocation_48(self):
        alloc = compute_job_item_allocation(68)
        self.assertGreaterEqual(alloc["ARRAY_ITEMS"] * alloc["JOBS_PER_ITEM"], 48)
