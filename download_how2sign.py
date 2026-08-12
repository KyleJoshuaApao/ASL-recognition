import os

def download_how2sign_translations():
    print("How2Sign Dataset Download Tool (Mock Mode)")
    print("==========================================")
    print("Simulating the download of How2Sign translation files for demonstration.")
    
    output_dir = "how2sign_data"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create mock data for train, val, and test
    splits = {
        "train": [
            "vid123\tA man is walking down the street.",
            "vid124\tShe is signing about the weather."
        ],
        "val": [
            "vid200\tWelcome to the continuous sign language dataset."
        ],
        "test": [
            "vid300\tThis is a test sentence for the database."
        ]
    }
    
    for split, lines in splits.items():
        file_path = os.path.join(output_dir, f"{split}_translations.txt")
        print(f"Generating mock data for {split} -> {file_path}")
        with open(file_path, 'w', encoding='utf-8') as f:
            for line in lines:
                f.write(line + "\n")
                
    print("Mock download complete! You can now run ingest_how2sign_to_sql.py")

if __name__ == '__main__':
    download_how2sign_translations()
