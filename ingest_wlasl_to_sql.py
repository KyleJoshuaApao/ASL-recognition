import json
import os
from db_models import get_session, Gloss, Video, init_db

def ingest_wlasl():
    json_path = "WLASL_v0.3.json"
    
    if not os.path.exists(json_path):
        print(f"Dataset file {json_path} not found. Please run download_dataset.py first to download the JSON, or ensure the file exists.")
        return
        
    print("Initializing Database...")
    init_db()
    
    session = get_session()
    
    print("Loading WLASL JSON...")
    with open(json_path, 'r') as f:
        data = json.load(f)
        
    print("Ingesting data into SQL Server...")
    for entry in data:
        gloss_word = entry['gloss']
        
        # Check if gloss already exists
        gloss = session.query(Gloss).filter_by(word=gloss_word).first()
        if not gloss:
            gloss = Gloss(word=gloss_word)
            session.add(gloss)
            session.commit()
            
        for inst in entry['instances']:
            video_url = inst['url']
            vid_id = inst['video_id']
            
            # Check if video already exists
            video = session.query(Video).filter_by(video_id=vid_id).first()
            if not video:
                # We skip youtube videos in the download script, but we can still store their metadata in the DB
                video = Video(
                    gloss_id=gloss.id,
                    video_id=vid_id,
                    url=video_url,
                    status='PENDING'
                )
                session.add(video)
                
    session.commit()
    print("Ingestion complete.")

if __name__ == '__main__':
    ingest_wlasl()
