#!/usr/bin/env python3
import sys
import subprocess
import zipfile
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DESKTOP = Path.home() / 'Desktop'
REQUIRED_DIRS = ['data', 'models', 'src', 'ui']
UI_FILES = ['index.html', 'style.css', 'script.js']
ZIP_MAP = {
    'processed_stemming.zip': ROOT / 'data' / 'processed_stemming',
    'processed_lemmatization.zip': ROOT / 'data' / 'processed_lemmatization',
    'tfidf_models.zip': ROOT / 'models',
}
ARCHIVE_DIR = ROOT / 'archives'
DEPENDENCIES = ['nltk', 'scikit-learn', 'scipy', 'numpy', 'flask']


def ensure_directories():
    print('Creating required directories...')
    for name in REQUIRED_DIRS + ['archives']:
        path = ROOT / name
        path.mkdir(parents=True, exist_ok=True)
        print(f'  - {path}')


def copy_ui_files_from_desktop():
    print('Copying UI files from Desktop into project ui/ folder...')
    ui_dir = ROOT / 'ui'
    for name in UI_FILES:
        desktop_file = DESKTOP / name
        target = ui_dir / name
        if desktop_file.exists():
            print(f'  - Copying {desktop_file} -> {target}')
            shutil.copy2(str(desktop_file), str(target))
        elif target.exists():
            print(f'  - ui/{name} already exists, skipping copy')
        else:
            print(f'  - Warning: {name} not found on Desktop and no ui/{name} exists')


def move_ui_files():
    print('Moving UI files into ui/ directory if needed...')
    ui_dir = ROOT / 'ui'
    for name in UI_FILES:
        source = ROOT / name
        target = ui_dir / name
        if source.exists() and not target.exists():
            print(f'  - Moving {source.name} to ui/{source.name}')
            shutil.move(str(source), str(target))
        elif target.exists():
            print(f'  - ui/{source.name} already exists, skipping move')
        else:
            print(f'  - {source.name} not found in project root')


def find_zip_file(filename):
    if DESKTOP.exists():
        candidate = DESKTOP / filename
        if candidate.exists():
            return candidate
        for path in DESKTOP.rglob(filename):
            return path
    return None


def copy_zip_to_project(zip_path: Path, destination: Path):
    destination.parent.mkdir(parents=True, exist_ok=True)
    target_path = destination.parent / zip_path.name
    if not target_path.exists():
        print(f'  - Copying {zip_path.name} to {target_path}')
        shutil.copy2(str(zip_path), str(target_path))
    else:
        print(f'  - Archive {target_path.name} already exists in project')
    return target_path


def extract_zip(zip_path: Path, destination: Path):
    if not zip_path or not zip_path.exists():
        print(f'  - ZIP file not found: {zip_path}')
        return False
    destination.mkdir(parents=True, exist_ok=True)
    print(f'  - Extracting {zip_path.name} -> {destination}')
    with zipfile.ZipFile(zip_path, 'r') as archive:
        archive.extractall(str(destination))
    return True


def setup_archives():
    print('Copying and unpacking archive files from Desktop...')
    archive_root = ARCHIVE_DIR
    for archive_name, destination in ZIP_MAP.items():
        zip_path = find_zip_file(archive_name)
        if zip_path is None:
            print(f'  - Warning: {archive_name} not found on Desktop')
            continue
        project_zip = copy_zip_to_project(zip_path, archive_root / archive_name)
        extract_zip(project_zip, destination)


def create_virtualenv():
    venv_dir = ROOT / 'venv'
    if not venv_dir.exists():
        print('Creating virtual environment in venv/...')
        subprocess.run([sys.executable, '-m', 'venv', str(venv_dir)], check=True)
    else:
        print('Virtual environment already exists at venv/')
    return venv_dir / 'bin' / 'python'


def install_packages(python_exe: Path):
    print('Installing Python dependencies in virtual environment...')
    subprocess.run([str(python_exe), '-m', 'pip', 'install', '--upgrade', 'pip'], check=True)
    subprocess.run([str(python_exe), '-m', 'pip', 'install'] + DEPENDENCIES, check=True)


def download_nltk(python_exe: Path):
    print('Downloading NLTK data (punkt, stopwords, wordnet, omw-1.4). This requires internet access once.')
    subprocess.run([
        str(python_exe),
        '-c',
        'import nltk; nltk.download("punkt"); nltk.download("stopwords"); nltk.download("wordnet"); nltk.download("omw-1.4")'
    ], check=True)


def main():
    ensure_directories()
    move_ui_files()
    setup_archives()
    python_exe = create_virtualenv()
    install_packages(python_exe)
    download_nltk(python_exe)
    print('\nSetup complete!')
    print('Run the backend with:')
    print('  venv/bin/python app.py')


if __name__ == '__main__':
    main()
