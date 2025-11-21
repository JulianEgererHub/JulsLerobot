
from huggingface_hub import HfApi

hub_api = HfApi()
hub_api.create_tag("JulianEgerer/my_first_dataset_corrected", tag="v3.0" , repo_type="dataset")
