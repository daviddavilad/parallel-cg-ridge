from parallel_cg_ridge.cg_mpi import row_counts

def test_row_counts_cover_all_rows():
    for n, size in [(17, 4), (500, 8), (500000, 32), (10, 10)]:
        counts, offsets = row_counts(n, size)
        assert counts.sum() == n
        assert offsets[0] == 0
        assert all(offsets[i+1] == offsets[i] + counts[i] for i in range(size - 1))