import sys
import subprocess

def main():
    if len(sys.argv) < 2:
        print("Usage: python dispatcher.py [process_name] [additional_args]")
        sys.exit(1)

    process_name = sys.argv[1]

    if process_name == 'slide_preprocessing':
        subprocess.run(['python', 'extract_multiple_wsi.py'] + sys.argv[2:])
    elif process_name == 'plip_preprocessing':
        subprocess.run(['python', 'process_tiles.py'] + sys.argv[2:])
    elif process_name == 'train_genomic_plip':
        subprocess.run(['python', 'train_genomic_plip_model.py'] + sys.argv[2:])
    elif process_name == 'train_test_split':
        subprocess.run(['python', 'train_test_split.py'] + sys.argv[2:])
    elif process_name == 'train_test_split_gplip':
        subprocess.run(['python', 'train_test_split_GPLIP_tile_features.py'] + sys.argv[2:])
    elif process_name == 'train_gwsif_classifier':
        subprocess.run(['python', 'train_GWSIF_classifier.py'] + sys.argv[2:])
    elif process_name == 'train_f_classifier':
        subprocess.run(['python', 'train_F_classifier.py'] + sys.argv[2:])
    elif process_name == 'extract_genomic_aligned_tile_features':
        subprocess.run(['python', 'extract_genomic_aligned_tiles_features.py'] + sys.argv[2:])
    elif process_name == 'jupyter_lab':
        subprocess.run(['jupyter', 'lab', '--ip=0.0.0.0', '--no-browser', '--allow-root'])
    else:
        print(f"Unknown process: {process_name}")
        sys.exit(1)

if __name__ == '__main__':
    main()
