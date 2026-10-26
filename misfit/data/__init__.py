"""Dataset loaders and adapters for the multi-site reliability experiments.

Modules:
    mri_multisite: MRIData container, CSV adapter, OOD/transfer splits,
        multi-environment conversion, and a synthetic multi-site generator.
    abide_loader: ABIDE-I resting-state fMRI loader (nilearn fetcher).
    oasis_loader: OASIS-1 VBM gray-matter loader (nilearn fetcher).
    ukbb_loader: UK Biobank structural MRI loader (local share).

All loaders return the MRIData container defined in mri_multisite, with
per-sample features, targets, subject ids, and site/environment ids.
"""
