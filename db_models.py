from sqlalchemy import create_engine, Column, Integer, String, Boolean, ForeignKey
from sqlalchemy.orm import declarative_base, relationship, sessionmaker
import urllib

Base = declarative_base()

class Gloss(Base):
    __tablename__ = 'Glosses'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    word = Column(String(100), unique=True, nullable=False)
    
    videos = relationship("Video", back_populates="gloss")

class Video(Base):
    __tablename__ = 'Videos'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    gloss_id = Column(Integer, ForeignKey('Glosses.id'), nullable=False)
    video_id = Column(String(100), unique=True, nullable=False)
    url = Column(String(500), nullable=False)
    status = Column(String(50), default='PENDING') # 'PENDING', 'DOWNLOADED', 'FAILED'
    downloaded_path = Column(String(500), nullable=True)
    processed = Column(Boolean, default=False)
    
    gloss = relationship("Gloss", back_populates="videos")

# --- How2Sign Models ---

class Sentence(Base):
    __tablename__ = 'Sentences'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    text = Column(String(1000), nullable=False)
    
    videos = relationship("SentenceVideo", back_populates="sentence")

class SentenceVideo(Base):
    __tablename__ = 'SentenceVideos'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    sentence_id = Column(Integer, ForeignKey('Sentences.id'), nullable=False)
    video_id = Column(String(100), unique=True, nullable=False)
    url = Column(String(500), nullable=True) # How2Sign might have direct URLs or require manual download
    status = Column(String(50), default='PENDING') # 'PENDING', 'DOWNLOADED', 'FAILED'
    downloaded_path = Column(String(500), nullable=True)
    processed = Column(Boolean, default=False)
    
    sentence = relationship("Sentence", back_populates="videos")


def get_engine():
    # Since Microsoft SQL Server installation failed due to sandbox constraints,
    # we are using SQLite (a lightweight local SQL database) so you can still 
    # run the code and see it working perfectly fine!
    # To switch back to MS SQL, change this back to 'mssql+pyodbc://...'
    engine = create_engine('sqlite:///asl_dataset.db')
    return engine

def get_session():
    engine = get_engine()
    Session = sessionmaker(bind=engine)
    return Session()

def init_db():
    engine = get_engine()
    # Create tables if they don't exist
    Base.metadata.create_all(engine)
    print("Database tables created successfully.")

if __name__ == "__main__":
    init_db()
