"""CG-NAFNet (Cluster-Guided Dynamic NAFNet).

Single-pass, real-time image restoration for images corrupted by an
unknown-order composition of speckle noise, Gaussian noise, and downsampling.
Soft cluster-guided FiLM conditioning routes restoration behaviour per stage.

Implementation spec:      implementation.md (repo root)
Architectural rationale:  clusir_nafnet_restoration_design.md (if present)

Run order:
  pytest cgnafnet/tests                 # unit tests (phases 1-2)
  python -m cgnafnet.train --config cgnafnet/configs/base.yaml
  python -m cgnafnet.validate_clusters --config cgnafnet/configs/base.yaml
  python -m cgnafnet.benchmark_latency --config cgnafnet/configs/base.yaml
"""