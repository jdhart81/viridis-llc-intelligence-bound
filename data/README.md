# Data Sources

This directory contains data sources and documentation for numerical estimates used in the paper.

## Biosphere Information Estimates (Table III)

### Human Genome
- **Size**: 3.2 × 10⁹ base pairs
- **Information**: ~6.4 × 10⁹ bits (2 bits per base pair)
- **Source**: Human Genome Project, NCBI Reference Sequence

### Total Species Genomes
- **Estimated species**: ~10⁷ eukaryotic species
- **Average genome size**: ~10⁸ bits
- **Unique information**: ~10¹⁵ bits (accounting for phylogenetic redundancy)
- **Sources**: 
  - Mora, C., et al. (2011). "How Many Species Are There on Earth and in the Ocean?" PLoS Biology.
  - International Barcode of Life project (iBOL)

### Ecological Interaction Networks
- **Estimated information**: ~10¹² bits
- **Source**: Estimates based on food web complexity data
  - Dunne, J. A., et al. (2002). "Network structure and biodiversity loss in food webs." Ecology Letters.

### Microbiome Diversity
- **Estimated information**: ~10¹⁴ bits
- **Sources**:
  - Earth Microbiome Project
  - Locey, K. J., & Lennon, J. T. (2016). "Scaling laws predict global microbial diversity." PNAS.

### Internet Comparison
- **Total internet data**: ~100 ZB (zettabytes) = 8 × 10²³ bits
- **Unique high-quality content**: ~10¹⁴ bits (conservative estimate)
- **Sources**:
  - IDC Global DataSphere Forecast
  - Wikipedia total text size: ~10¹¹ bits

## System Parameters (Table II)

### Human Brain
- **Power consumption**: 20 W (metabolic, approximately 20% of resting metabolism)
- **Temperature**: 310 K (37°C body temperature)
- **Sensory bandwidth**: ~10⁹ bits/s
  - Retina: ~10⁷ bits/s (Koch, K., et al., 2006)
  - Other senses: ~10⁸ bits/s
- **Estimated D**: 10⁻³ (most sensory input is predictable/redundant)
- **Source**: 
  - Attwell, D., & Laughlin, S. B. (2001). "An energy budget for signaling in the grey matter of the brain." Journal of Cerebral Blood Flow & Metabolism.

### NVIDIA H100
- **TDP**: 700 W
- **Operating temperature**: ~350 K (77°C typical)
- **HBM3 bandwidth**: 3.35 TB/s = 2.68 × 10¹³ bits/s
- **Estimated D for internet text**: 0.05
- **Source**: NVIDIA H100 Tensor Core GPU Datasheet

## Biodiversity-D Coupling Model

The cascade model `D_after = D₀ × (1-f)^(1+βf)` is illustrative rather than empirically calibrated. It captures the qualitative observation that:

1. Ecosystem resilience initially buffers against biodiversity loss
2. Above a threshold, cascade failures cause accelerating collapse

Empirical calibration would require ecosystem-specific data from biodiversity manipulation experiments.

## Uncertainty

All estimates in Table III carry ±1-2 orders of magnitude uncertainty. The key conclusions of the paper are robust to this uncertainty because:

1. The data-limited vs. power-limited finding has 10¹²-10¹⁵× headroom
2. The Gaia-Intelligence Proposition depends on D_bio > D_synthetic, not absolute values
3. Strategy comparison results hold for any reasonable D(t) dynamics

## Data Availability

The simulation code and data that support the findings of this study are contained in this repository. No external datasets are required to reproduce the numerical results.

For empirical validation of the D measurement protocols, we recommend:
- Acoustic recordings from intact vs. degraded ecosystems
- Camera trap time series
- Environmental sensor data (temperature, humidity, etc.)

These datasets should be processed using the `compression_estimator` function from `code/estimators.py`.
