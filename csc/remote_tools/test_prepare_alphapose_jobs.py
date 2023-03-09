import unittest

from prepare_alphapose_jobs import compute_job_item_allocation


class TestJobAllocation (unittest.TestCase):
    def test_compute_job_item_allocation_64(self):
        alloc = compute_job_item_allocation(64, pref_array_jobs=4)
        self.assertEqual(alloc["ARRAY_ITEMS"], 16)
        self.assertEqual(alloc["JOBS_PER_ITEM"], 4)

    def test_compute_job_item_allocation_no_empty(self):
        """Test last array item gets at least 1 job.
        """
        for n in range(1, 80):
            alloc = compute_job_item_allocation(n, pref_array_jobs=4)
            n_ai = alloc["ARRAY_ITEMS"]
            n_jbi = alloc["JOBS_PER_ITEM"]
            self.assertLessEqual(n_ai * n_jbi - n, n_jbi)

    def test_compute_job_item_allocation_no_missing(self):
        for n in range(1, 68):
            alloc = compute_job_item_allocation(n, pref_array_jobs=4)
            self.assertGreaterEqual(alloc["ARRAY_ITEMS"] * alloc["JOBS_PER_ITEM"], n)

    def test_compute_job_item_allocation_48(self):
        alloc = compute_job_item_allocation(48, pref_array_jobs=4)
        self.assertGreaterEqual(
            alloc["ARRAY_ITEMS"] * alloc["JOBS_PER_ITEM"], 48)
