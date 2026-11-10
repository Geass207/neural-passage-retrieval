from src.data.base_dataset import BaseDataset


def get_dataset(name: str, data_dir: str = "data/", subset_pct: float = None, seed: int = 42) -> BaseDataset:
    """
    Return a dataset instance by name.
    To add a new dataset: implement BaseDataset and register it here.

    Args:
        name: Dataset name ("msmarco", "msmarco_subset", or any BEIR dataset)
        data_dir: Root data directory
        subset_pct: For "msmarco_subset", percentage to sample (0.1 = 10%, 0.05 = 5%)
        seed: Random seed for reproducibility

    Examples:
        # Full MS MARCO
        dataset = get_dataset("msmarco", data_dir="data/")

        # 10% MS MARCO subset (faster for development)
        dataset = get_dataset("msmarco_subset", data_dir="data/", subset_pct=0.1)

        # 5% subset with different seed
        dataset = get_dataset("msmarco_subset", data_dir="data/", subset_pct=0.05, seed=123)
    """
    if name == "msmarco":
        from src.data.loaders.msmarco_loader import MSMarcoLoader
        return MSMarcoLoader(data_dir=data_dir)

    if name == "msmarco_subset":
        pct = subset_pct or 0.1  # Default to 10% if not specified
        from src.data.loaders.msmarco_subset_loader import MSMarcoSubsetLoader
        return MSMarcoSubsetLoader(data_dir=data_dir, subset_pct=pct, seed=seed)

    # Generic BEIR loader for any other dataset present in data_dir/<name>/
    from src.data.loaders.beir_loader import BEIRLoader
    return BEIRLoader(name=name, data_dir=data_dir)
