"""Module to download and load pre-trained ALIGNN models."""
import requests
import os
import zipfile
from tqdm import tqdm
from alignn.models.alignn import ALIGNN, ALIGNNConfig
import tempfile
import torch

tqdm.pandas()

def get_figshare_model(model_name="mp_e_form_alignnn",device=torch.device('cuda')):
    """Get ALIGNN torch models from figshare."""
    # https://figshare.com/projects/ALIGNN_models/126478
    all_models = {
        "jv_formation_energy_peratom_alignn": [
            "https://figshare.com/ndownloader/files/31458679",
            1,
        ],
        "mp_e_form_alignnn": [
            "https://figshare.com/ndownloader/files/31458811",
            1,
        ],
    }
    tmp = all_models[model_name]
    url = tmp[0]
    output_features = tmp[1]
    if len(tmp) > 2:
        config_params = tmp[2]
    else:
        config_params = {}
    zfile = model_name + ".zip"
    path = str(os.path.join(os.path.dirname(__file__), zfile))
    if not os.path.isfile(path):
        response = requests.get(url, stream=True)
        total_size_in_bytes = int(response.headers.get("content-length", 0))
        block_size = 1024  # 1 Kibibyte
        progress_bar = tqdm(
            total=total_size_in_bytes, unit="iB", unit_scale=True
        )
        with open(path, "wb") as file:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)
        progress_bar.close()
    zp = zipfile.ZipFile(path)
    names = zp.namelist()
    chks = []
    for i in names:
        if "checkpoint_" in i and "pt" in i:
            tmp = i
            chks.append(i)
    data = zipfile.ZipFile(path).read(tmp)
    model = ALIGNN(
        ALIGNNConfig(
            name="alignn", output_features=output_features, **config_params
        )
    )
    _, filename = tempfile.mkstemp()
    with open(filename, "wb") as f:
        f.write(data)
    model.load_state_dict(torch.load(filename, map_location=device)["model"])
    model.to(device)
    model.eval()
    return model