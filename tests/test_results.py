from parallel_cg_ridge.results import efficiency, speedup

def test_speedup_is_efficiency_times_p():
    rows = [
        {"n": 100, "P": 1, "wall": 8.0, "iters": 90},
        {"n": 100, "P": 2, "wall": 4.5, "iters": 91},
        {"n": 100, "P": 4, "wall": 2.4, "iters": 89},
    ]
    for normalize in (True, False):
        s = speedup(rows, normalize=normalize)[100]
        e = efficiency(rows, normalize=normalize)[100]
        for (p, sp), (_, ef) in zip(s, e):
            assert abs(sp / p - ef) < 1e-12