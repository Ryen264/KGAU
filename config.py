import yaml
import logging
import subprocess
import torch
import os
import time

_config = None # type: ConfigDict
class ConfigDict(dict):
    __getattr__ = dict.__getitem__

def config(config_path: str) -> ConfigDict:
    """
    default: config("config_wn18rr.yaml")
    """
    def _make_config_dict(obj: dict) -> ConfigDict:
        if isinstance(obj, dict):
            return ConfigDict({k: _make_config_dict(v) for k, v in obj.items()})
        elif isinstance(obj, list):
            return [_make_config_dict(x) for x in obj]
        else:
            return obj
    
    global _config
    if _config is None:
        with open(config_path) as f:
            _config = _make_config_dict(yaml.load(f, Loader=yaml.FullLoader))
    return _config

def overwrite_config_with_args(args: list=[], sep: str='.') -> None:
    """
    Manually pass parameters. E.g. overwrite_config_with_args(["--pretrain_config=TransD"])
    TransE.n_epoch=2
    steps=["TransE", "n_epoch"]
    steps[:-1] = ["TransE"]
    steps[-1] = "n_epoch"
    val=2
    """
    def path_set(path: str, val: str, sep: str='.', auto_convert: bool=False) -> None:
        steps = path.split(sep)
        obj = _config
        for step in steps[:-1]:
            obj = obj[step]
        old_val = obj[steps[-1]]
        
        if not auto_convert:
            obj[steps[-1]] = val
        elif isinstance(old_val, bool):
            obj[steps[-1]] = val.lower() == 'true'
        elif isinstance(old_val, float):
            obj[steps[-1]] = float(val)
        elif isinstance(old_val, int):
            try:
                obj[steps[-1]] = int(val)
            except ValueError:
                obj[steps[-1]] = float(val)
        else:
            obj[steps[-1]] = val
    
    for arg in args:
        if arg.startswith('--') and '=' in arg:
            path, val = arg[2:].split('=')
            if path != 'config':
                path_set(path, val, sep, auto_convert=True)

def dump_config() -> None:
    def _dump_config(obj: dict, prefix: tuple) -> None:
        if isinstance(obj, dict):
            for k, v in obj.items():
                _dump_config(v, prefix + (k,))
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                _dump_config(v, prefix + (str(i),))
        else:
            logging.debug('%s=%s', '.'.join(prefix), repr(obj))
    return _dump_config(_config, tuple())

def select_gpu() -> int:
    if not torch.cuda.is_available():
        logging.warning("No GPU available. Running on CPU.")
        return None

    try:
        nvidia_info = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        logging.warning("nvidia-smi not found or failed. Running on CPU.")
        return None

    gpu_info = False
    gpu_info_line = 0
    proc_info = False
    gpu_mem = []
    gpu_occupied = set()

    for line in nvidia_info.stdout.split(b'\n'):
        line = line.decode().strip()
        if gpu_info:
            gpu_info_line += 1
            if line == '':
                gpu_info = False
                continue
            if gpu_info_line % 3 == 2:
                mem_info = line.split('|')[2]
                used_mem_mb = int(mem_info.strip().split()[0][:-3])
                gpu_mem.append(used_mem_mb)
        if proc_info:
            if line == '|  No running processes found                                                 |':
                continue
            if line == '+-----------------------------------------------------------------------------+':
                proc_info = False
                continue
            proc_gpu = int(line.split()[1])
            gpu_occupied.add(proc_gpu)
        if line == '|===============================+======================+======================|':
            gpu_info = True
        if line == '|=============================================================================|':
            proc_info = True

    if not gpu_mem:
        logging.info("Could not parse nvidia-smi output. Defaulting to GPU 0.")
        return 0

    for i in range(len(gpu_mem)):
        if i not in gpu_occupied:
            logging.info('Automatically selected GPU %d because it is vacant.', i)
            return i
    for i in range(len(gpu_mem)):
        if gpu_mem[i] == min(gpu_mem):
            logging.info('All GPUs are occupied. Automatically selected GPU %d because it has the most free memory.', i)
            return i
        
def set_device(gpu_id: int) -> torch.device:
    def _cpu_fallback(reason: str) -> torch.device:
        logging.warning("%s Falling back to CPU.", reason)
        return torch.device("cpu")

    if gpu_id is None or not torch.cuda.is_available():
        logging.info("No GPU available. Running on CPU.")
        return torch.device("cpu")

    try:
        torch.cuda.set_device(gpu_id)
        device = torch.device(f"cuda:{gpu_id}")
        dev_name = torch.cuda.get_device_name(gpu_id)
        capability = torch.cuda.get_device_capability(gpu_id)
        arch_list = torch.cuda.get_arch_list() if hasattr(torch.cuda, 'get_arch_list') else []
        required_arch = f"sm_{capability[0]}{capability[1]}"

        if arch_list and required_arch not in arch_list:
            return _cpu_fallback(
                "CUDA runtime mismatch: this PyTorch build does not contain kernels for "
                f"{required_arch}. Available arches={arch_list}, torch_cuda={torch.version.cuda}."
            )

        # Probe a tiny CUDA kernel early so incompatibilities surface here instead of deep in training.
        probe = torch.ones(1, device=device)
        probe = probe + 1
        if hasattr(torch.cuda, "synchronize"):
            torch.cuda.synchronize(device)

        logging.info(
            "Using GPU: %s | name=%s | capability=sm_%d%d | torch_cuda=%s | build_arches=%s",
            device,
            dev_name,
            capability[0],
            capability[1],
            torch.version.cuda,
            arch_list,
        )
        return device
    except Exception as exc:
        msg = str(exc)
        if 'no kernel image is available for execution on the device' in msg:
            return _cpu_fallback(
                "CUDA runtime mismatch: installed PyTorch/CUDA build does not support this GPU architecture. "
                f"gpu_id={gpu_id}, torch_cuda={torch.version.cuda}."
            )
        return _cpu_fallback(f"CUDA initialization failed: {msg}")


def build_timestamped_filename(prefix: str, ext: str) -> str:
    """Build filename as <prefix>_yymmdd-hhmmss<ext>."""
    ts = time.strftime("%y%m%d-%H%M%S")
    normalized_prefix = (prefix or "").rstrip("_")
    if not normalized_prefix:
        normalized_prefix = "training"
    return f"{normalized_prefix}_{ts}{ext}"

def logger_init() -> None:
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(module)15s %(asctime)s %(message)s', datefmt='%H:%M:%S'))
    root_logger.addHandler(console_handler)

    if (_config.log.to_file):
        log_dir = os.path.join('.', 'logs', _config.dataset, _config.task)
        os.makedirs(log_dir, exist_ok=True)
        log_basename = build_timestamped_filename(_config.log.prefix, ".log")
        log_filename = os.path.join(log_dir, log_basename)
        file_handler = logging.FileHandler(log_filename)
        file_handler.setFormatter(logging.Formatter('%(module)15s %(asctime)s %(message)s', datefmt='%H:%M:%S'))
        root_logger.addHandler(file_handler)

    if (_config.log.dump_config):
        dump_config()

def log_step(label: str, start_ts: float) -> float:
    """Print elapsed time for a pipeline step and return a new start timestamp."""
    elapsed = time.perf_counter() - start_ts
    print(f"[TIMER] {label}: {elapsed:.2f}s")
    return time.perf_counter()

gpu_id = select_gpu()
device = set_device(gpu_id)