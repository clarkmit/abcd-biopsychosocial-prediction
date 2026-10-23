Test-only stand-ins for `torch` and `tabpfn`.

The published notebook imports both at the top of cell 5. These stubs let the equivalence
tests execute it on machines without PyTorch/TabPFN; they are put on `sys.path` only when
the real packages are not installed, and only for the equivalence tests. The stand-in
TabPFN is a small linear model (`Ridge` / `LogisticRegression`). Across-category runs never
fit TabPFN; the within-category plan does, with the same stand-in on both sides, so the two
runs remain comparable.
