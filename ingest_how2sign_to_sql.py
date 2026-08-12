import os
from db_models import get_session, init_db, Sentence, SentenceVideo

def ingest_how2sign():
    # Expects format: video_id \t sentence
    # Note: the actual format of How2Sign translation files might differ slightly,
    # but it's typically tab-separated or space-separated with ID and text.
    
    data_dir = "how2sign_data"
    files = ["train_translations.txt", "val_translations.txt", "test_translations.txt"]
    
    print("Initializing Database for How2Sign...")
    init_db()
    session = get_session()
    
    for file_name in files:
        file_path = os.path.join(data_dir, file_name)
        if not os.path.exists(file_path):
            print(f"Skipping {file_name} (Not found)")
            continue
            
        print(f"Ingesting {file_name}...")
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                # Assume standard "video_id \t translation text"
                parts = line.split('\t', 1)
                if len(parts) != 2:
                    # Try space split if tab fails
                    parts = line.split(' ', 1)
                    if len(parts) != 2:
                        continue
                        
                vid_id, text = parts[0], parts[1]
                
                # Insert Sentence
                sentence = session.query(Sentence).filter_by(text=text).first()
                if not sentence:
                    sentence = Sentence(text=text)
                    session.add(sentence)
                    session.commit()
                
                # Insert SentenceVideo mapping
                video = session.query(SentenceVideo).filter_by(video_id=vid_id).first()
                if not video:
                    video = SentenceVideo(
                        sentence_id=sentence.id,
                        video_id=vid_id,
                        status='PENDING' # Need video links to download
                    )
                    session.add(video)
                    
            session.commit()
            
    print("How2Sign Ingestion complete.")

if __name__ == '__main__':
    ingest_how2sign()
