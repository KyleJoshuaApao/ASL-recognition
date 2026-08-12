import os
import urllib.request
import urllib.error
from db_models import get_session, Gloss, Video

def download_wlasl():
    session = get_session()
    
    # We map the words we want to folder names
    target_words = {
        "hello": "hello",
        "thank you": "thanks",
        "love": "iloveyou"
    }
    
    raw_data_dir = "raw_data"
    os.makedirs(raw_data_dir, exist_ok=True)
    
    for gloss_word, folder_name in target_words.items():
        print(f"Checking videos for '{gloss_word}'...")
        gloss = session.query(Gloss).filter_by(word=gloss_word).first()
        if not gloss:
            print(f"Gloss '{gloss_word}' not found in database. Make sure you ingested the data first.")
            continue
            
        folder_path = os.path.join(raw_data_dir, folder_name)
        os.makedirs(folder_path, exist_ok=True)
        
        # Get videos that are PENDING and not youtube links
        videos = session.query(Video).filter(
            Video.gloss_id == gloss.id,
            Video.status == 'PENDING',
            ~Video.url.ilike('%youtube%')
        ).limit(5).all() # download up to 5 videos per class
        
        downloaded_count = 0
        for video in videos:
            if downloaded_count >= 5:
                break
                
            try:
                print(f"Attempting to download {gloss_word} from {video.url}")
                req = urllib.request.Request(video.url, headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(req) as response:
                    file_path = os.path.join(folder_path, f"{video.video_id}.mp4")
                    with open(file_path, 'wb') as out_file:
                        out_file.write(response.read())
                
                print(f"  -> Successfully downloaded {video.video_id}.mp4")
                
                # Update database record
                video.status = 'DOWNLOADED'
                video.downloaded_path = file_path
                session.commit()
                
                downloaded_count += 1
            except Exception as e:
                print(f"  -> Failed: {e}")
                video.status = 'FAILED'
                session.commit()
                
    print("Dataset download via SQL complete.")

if __name__ == '__main__':
    download_wlasl()
