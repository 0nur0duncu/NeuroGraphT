import subprocess
import sys
from pathlib import Path
from typing import Optional

try:
    import boto3
    from botocore import UNSIGNED
    from botocore.config import Config
except ImportError:
    print("boto3 kuruluyor...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "boto3", "-q"])
    import boto3
    from botocore import UNSIGNED
    from botocore.config import Config

try:
    from tqdm import tqdm
except ImportError:
    print("tqdm kuruluyor...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "tqdm", "-q"])
    from tqdm import tqdm

BUCKET = "physionet-open"
PREFIX = "sleep-edfx/1.0.0/"

def get_dataset_path(base_dir: Optional[str] = None) -> Path:
    """
    Veri seti için uygun path'i döndürür.
    Colab (/content) veya yerel ortam (relative path) için otomatik seçim.
    """
    if base_dir:
        return Path(base_dir)
    
    colab_path = Path("/content/dataset/sleep-edfx")
    if Path("/content").exists():
        return colab_path
    
    return Path("dataset/sleep-edfx")

def check_dataset_exists(data_dir: Path, study: str = 'SC', min_files: int = 10) -> bool:
    """
    Veri setinin mevcut olup olmadığını kontrol eder.
    """
    if not data_dir.exists():
        return False
    
    study_dir = data_dir / ("sleep-cassette" if study == 'SC' else "sleep-telemetry")
    prefix = 'SC' if study == 'SC' else 'ST'
    
    search_dirs = [study_dir, data_dir]
    
    for search_dir in search_dirs:
        if not search_dir.exists():
            continue
        
        edf_files = [f for f in search_dir.glob(f'{prefix}*.edf') if 'Hypnogram' not in f.name]
        hypno_files = list(search_dir.glob(f'{prefix}*Hypnogram.edf'))
        
        if len(edf_files) >= min_files and len(hypno_files) >= min_files:
            return True
    
    return False

def ensure_dataset(
    data_dir: Optional[str] = None,
    study: str = 'SC',
    force_download: bool = False,
    verbose: bool = True
) -> Path:
    """
    Veri setinin mevcut olduğundan emin olur, yoksa indirir.
    
    Args:
        data_dir: Veri seti dizini (None ise otomatik seçilir)
        study: SC veya ST
        force_download: True ise mevcut veri setini siler ve yeniden indirir
        verbose: İlerleme mesajlarını göster
    
    Returns:
        Veri seti dizini (Path)
    """
    output_dir = get_dataset_path(data_dir)
    
    if not force_download and check_dataset_exists(output_dir, study):
        if verbose:
            print(f"✓ Veri seti mevcut: {output_dir}")
        return output_dir
    
    if verbose:
        print(f"Veri seti indiriliyor: {output_dir}")
        if force_download:
            print("(force_download=True, yeniden indiriliyor)")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))
        
        paginator = s3.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=BUCKET, Prefix=PREFIX)
        
        files = []
        for page in pages:
            for obj in page.get('Contents', []):
                files.append(obj)
        
        if verbose:
            iterator = tqdm(files, desc="İndiriliyor")
        else:
            iterator = files
        
        for obj in iterator:
            key = obj['Key']
            relative_path = key[len(PREFIX):]
            
            if not relative_path:
                continue
            
            local_path = output_dir / relative_path
            
            if local_path.exists():
                continue
                
            local_path.parent.mkdir(parents=True, exist_ok=True)
            
            s3.download_file(BUCKET, key, str(local_path))
        
        if verbose:
            print(f"\n✓ İndirme tamamlandı: {output_dir}")
            print(f"Dosya yapısı:")
            for item in output_dir.rglob('*.edf'):
                print(f"  {item.relative_to(output_dir)}")
        
        return output_dir
        
    except Exception as e:
        print(f"\n✗ Hata oluştu: {e}")
        raise

if __name__ == "__main__":
    ensure_dataset(verbose=True)
