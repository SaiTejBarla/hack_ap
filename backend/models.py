from sqlalchemy import Column, Integer, String, Text, DateTime
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

Base = declarative_base()

class AdvisoryLog(Base):
    __tablename__ = "advisory_logs"

    id = Column(Integer, primary_key=True, index=True)
    image_path = Column(String, nullable=True)
    prediction = Column(String, nullable=False)
    recommendation = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)

class VoiceLog(Base):
    __tablename__ = "voice_logs"

    id = Column(Integer, primary_key=True, index=True)
    voice_text = Column(Text, nullable=False)
    prediction = Column(String, nullable=False)
    recommendation = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
