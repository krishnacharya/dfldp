from pathlib import Path

def project_root():
    current_dir = Path(__file__).absolute().parent.parent
    return current_dir

def data_root():
    res = project_root()/"data"
    res.mkdir(parents=True, exist_ok=True)
    return res

def raw_data_root():
    res = data_root() / "raw"
    res.mkdir(parents=True, exist_ok=True)
    return res

def processed_data_root() -> Path:
    res = data_root() / "processed"
    res.mkdir(parents=True, exist_ok=True)
    return res

def output_dir():
    res = project_root() / "outputs"
    res.mkdir(parents=True, exist_ok=True)
    return res

def output_dir_name(name:str):
    res = project_root() / "outputs" / name
    res.mkdir(parents=True, exist_ok=True)
    return res

def src_dir():
    return project_root() / "src"
