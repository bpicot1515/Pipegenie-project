# tools/metabase_manager.py

import argparse
import zipfile
import shutil
from pathlib import Path
import json

# --- CONFIGURATION ---
# Define the default path where pipegenie looks for the metabase.
# This should match the default in your pipegenie code.
DEFAULT_METABASE_PATH = Path("./openml_metabase")

def export_metabase(output_path: Path, metabase_path: Path):
    """
    Exports the metabase by compressing it into a single zip file.
    """
    if not metabase_path.is_dir() or not any(metabase_path.iterdir()):
        print(f"❌ Error: Meta-base directory '{metabase_path}' not found or is empty.")
        return

    # Create a manifest file with metadata about the export
    manifest = {
        "export_date": datetime.now().isoformat(),
        "num_files": len(list(metabase_path.glob('*.json'))),
        "source_path": str(metabase_path.resolve()),
        # You could add pipegenie version info here in the future
    }

    try:
        with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # Add the manifest file
            zipf.writestr("manifest.json", json.dumps(manifest, indent=4))
            
            # Add all JSON files from the metabase
            for file_path in metabase_path.glob('*.json'):
                # The second argument is the path inside the zip file
                zipf.write(file_path, arcname=file_path.name)
        
        print(f"✅ Meta-base successfully exported to: {output_path}")

    except Exception as e:
        print(f"❌ Error during export: {e}")


def import_metabase(input_path: Path, metabase_path: Path):
    """
    Imports a metabase from a zip file, overwriting the existing one.
    """
    if not input_path.is_file():
        print(f"❌ Error: Input file '{input_path}' not found.")
        return

    if metabase_path.exists():
        backup_path = metabase_path.with_suffix(f'.bak.{datetime.now().strftime("%Y%m%d%H%M%S")}')
        print(f"ℹ️  Found existing meta-base at '{metabase_path}'.")
        print(f"    Moving it to '{backup_path}' as a backup.")
        shutil.move(str(metabase_path), str(backup_path))
    
    metabase_path.mkdir(parents=True, exist_ok=True)

    try:
        with zipfile.ZipFile(input_path, 'r') as zipf:
            # Verify it's a valid metabase archive by checking for manifest.json
            if 'manifest.json' not in zipf.namelist():
                print("❌ Error: This does not appear to be a valid meta-base archive (missing manifest.json).")
                # Restore backup if we created one
                if 'backup_path' in locals():
                    shutil.move(str(backup_path), str(metabase_path))
                return

            zipf.extractall(metabase_path)
        
        num_files_imported = len(list(metabase_path.glob('*.json')))
        print(f"✅ Meta-base successfully imported with {num_files_imported} entries.")
        print(f"   You can now run pipegenie using the path: '{metabase_path}'")

    except zipfile.BadZipFile:
        print(f"❌ Error: '{input_path}' is not a valid zip file.")
    except Exception as e:
        print(f"❌ Error during import: {e}")


if __name__ == "__main__":
    from datetime import datetime

    parser = argparse.ArgumentParser(
        description="""
        A tool to manage the pipegenie meta-learning knowledge base.
        Allows for exporting to a portable file and importing from one.
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", required=True, help="Available commands")

    # --- Export Command ---
    parser_export = subparsers.add_parser("export", help="Export the meta-base to a .zip file.")
    parser_export.add_argument(
        "-o", "--output",
        type=Path,
        required=True,
        help="Path for the output .zip file (e.g., metabase_export.zip)."
    )
    parser_export.add_argument(
        "--path",
        type=Path,
        default=DEFAULT_METABASE_PATH,
        help=f"Path to the source meta-base directory (default: {DEFAULT_METABASE_PATH})."
    )

    # --- Import Command ---
    parser_import = subparsers.add_parser("import", help="Import a meta-base from a .zip file.")
    parser_import.add_argument(
        "-i", "--input",
        type=Path,
        required=True,
        help="Path of the .zip file to import."
    )
    parser_import.add_argument(
        "--path",
        type=Path,
        default=DEFAULT_METABASE_PATH,
        help=f"Path to the destination meta-base directory (default: {DEFAULT_METABASE_PATH}). This will be overwritten."
    )

    args = parser.parse_args()

    if args.command == "export":
        export_metabase(output_path=args.output, metabase_path=args.path)
    elif args.command == "import":
        import_metabase(input_path=args.input, metabase_path=args.path)