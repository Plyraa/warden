import os
import json
from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    Float,
    Boolean,
    DateTime,
    ForeignKey,
    Text,
)
from sqlalchemy.orm import sessionmaker, declarative_base, relationship
from sqlalchemy.sql import func
import datetime

DATABASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "database")
if not os.path.exists(DATABASE_DIR):
    os.makedirs(DATABASE_DIR)

DATABASE_URL = f"sqlite:///{os.path.join(DATABASE_DIR, 'audio_analysis.db')}"

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class AudioAnalysis(Base):
    __tablename__ = "audio_analyses"

    id = Column(Integer, primary_key=True, index=True)
    original_filename = Column(String, unique=True, index=True, nullable=False)
    downsampled_filepath = Column(String, nullable=False)
    analysis_timestamp = Column(DateTime(timezone=True), server_default=func.now())

    avg_latency_ms = Column(Float)
    p10_latency_ms = Column(Float)
    p50_latency_ms = Column(Float)
    p90_latency_ms = Column(Float)

    ai_interrupting_user = Column(Boolean)
    user_interrupting_ai = Column(Boolean)
    talk_ratio = Column(Float)
    average_pitch_hz = Column(Float)
    words_per_minute = Column(Float)

    user_speech_segments_json = Column(String)  # JSON string
    agent_speech_segments_json = Column(String)  # JSON string
    agent_answer_latencies_ms_json = Column(String)  # JSON string

    # Relationship to Transcript
    transcript_id = Column(Integer, ForeignKey("transcripts.id"))
    transcript = relationship("Transcript", back_populates="audio_analysis")


class Transcript(Base):
    __tablename__ = "transcripts"

    id = Column(Integer, primary_key=True, index=True)
    original_filename = Column(String, unique=True, index=True, nullable=False)
    transcript_text = Column(Text)  # Full dialog
    word_level_transcript_json = Column(String)  # JSON string of word-level data
    has_overlaps = Column(Boolean, default=False)  # Flag for speech overlaps
    overlap_count = Column(Integer, default=0)  # Count of overlapping words
    transcript_metadata_json = Column(
        String, nullable=True
    )  # For additional ElevenLabs data
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationship back to AudioAnalysis (one-to-one)
    audio_analysis = relationship(
        "AudioAnalysis", back_populates="transcript", uselist=False
    )


def init_db():
    Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def add_analysis(db_session, metrics_data):
    user_windows_json = json.dumps(metrics_data.get("user_windows", []))
    agent_windows_json = json.dumps(metrics_data.get("agent_windows", []))
    agent_latencies_json = json.dumps(metrics_data.get("agent_answer_latencies", []))
    latency_metrics = metrics_data.get("latency_metrics", {})
    transcript_data = metrics_data.get("transcript_data")

    db_analysis = AudioAnalysis(
        original_filename=metrics_data["filename"],
        downsampled_filepath=metrics_data["downsampled_path"],
        analysis_timestamp=datetime.datetime.now(datetime.timezone.utc),
        avg_latency_ms=latency_metrics.get("avg_latency"),
        p10_latency_ms=latency_metrics.get("p10_latency"),
        p50_latency_ms=latency_metrics.get("p50_latency"),
        p90_latency_ms=latency_metrics.get("p90_latency"),
        ai_interrupting_user=metrics_data.get("ai_interrupting_user"),
        user_interrupting_ai=metrics_data.get("user_interrupting_ai"),
        talk_ratio=metrics_data.get("talk_ratio"),
        average_pitch_hz=metrics_data.get("average_pitch"),
        words_per_minute=metrics_data.get("words_per_minute"),
        user_speech_segments_json=user_windows_json,
        agent_speech_segments_json=agent_windows_json,
        agent_answer_latencies_ms_json=agent_latencies_json,
    )

    if transcript_data:
        # Create transcript record with extended information
        db_transcript = Transcript(
            original_filename=metrics_data["filename"],
            transcript_text=transcript_data["dialog"],
            word_level_transcript_json=json.dumps(transcript_data["words"]),
            has_overlaps=transcript_data.get("has_overlaps", False),
            overlap_count=transcript_data.get("overlap_count", 0),
            # Store any additional metadata from the transcript
            transcript_metadata_json=json.dumps(
                {
                    k: v
                    for k, v in transcript_data.items()
                    if k not in ["words", "dialog", "has_overlaps", "overlap_count"]
                }
            ),
        )
        db_analysis.transcript = db_transcript
        # db_session.add(db_transcript) # Handled by relationship cascade

    db_session.add(db_analysis)
    db_session.commit()
    db_session.refresh(db_analysis)
    return db_analysis


def get_analysis_by_filename(db_session, filename: str):
    return (
        db_session.query(AudioAnalysis)
        .filter(AudioAnalysis.original_filename == filename)
        .first()
    )


def recreate_metrics_from_db(db_record: AudioAnalysis):
    """Converts a DB record back to the metrics dictionary format."""
    if not db_record:
        return None

    metrics = {
        "filename": db_record.original_filename,
        "downsampled_path": db_record.downsampled_filepath,
        "latency_metrics": {
            "avg_latency": db_record.avg_latency_ms,
            "p10_latency": db_record.p10_latency_ms,
            "p50_latency": db_record.p50_latency_ms,
            "p90_latency": db_record.p90_latency_ms,
        },
        "agent_answer_latencies": json.loads(
            db_record.agent_answer_latencies_ms_json or "[]"
        ),
        "ai_interrupting_user": db_record.ai_interrupting_user,
        "user_interrupting_ai": db_record.user_interrupting_ai,
        "talk_ratio": db_record.talk_ratio,
        "average_pitch": db_record.average_pitch_hz,
        "words_per_minute": db_record.words_per_minute,
        "user_windows": json.loads(db_record.user_speech_segments_json or "[]"),
        "agent_windows": json.loads(db_record.agent_speech_segments_json or "[]"),
        "analysis_timestamp": db_record.analysis_timestamp,
    }

    if db_record.transcript:
        # Include all transcript data including overlap information
        transcript_data = {
            "dialog": db_record.transcript.transcript_text,
            "words": json.loads(
                db_record.transcript.word_level_transcript_json or "[]"
            ),
            "has_overlaps": db_record.transcript.has_overlaps,
            "overlap_count": db_record.transcript.overlap_count,
            "transcript_id": db_record.transcript.id,
        }

        # Add any additional metadata stored in transcript_metadata_json
        if db_record.transcript.transcript_metadata_json:
            try:
                additional_metadata = json.loads(
                    db_record.transcript.transcript_metadata_json
                )
                transcript_data.update(additional_metadata)
            except json.JSONDecodeError:
                pass

        metrics["transcript_data"] = transcript_data
    else:
        metrics["transcript_data"] = None
    return metrics


if __name__ == "__main__":
    # This allows creating the DB schema directly by running this file
    print("Initializing database...")
    init_db()
    print(f"Database initialized at {DATABASE_URL}")
    # Example usage (optional, for testing)
    # db = next(get_db())
    # test_metrics = {
    #     "filename": "test_file.mp3",
    #     "downsampled_path": "sampled_test_calls/test_file_downsampled.mp3",
    #     "latency_metrics": {"avg_latency": 100, "p10_latency": 50, "p50_latency": 100, "p90_latency": 150},
    #     "agent_answer_latencies": [50,100,150],
    #     "ai_interrupting_user": False,
    #     "user_interrupting_ai": True,
    #     "talk_ratio": 1.5,
    #     "average_pitch": 220.0,
    #     "words_per_minute": 150.0,
    #     "user_windows": [[0.5, 2.0], [3.0, 4.5]],
    #     "agent_windows": [[2.1, 2.9], [4.6, 5.5]],
    # }
    # add_analysis(db, test_metrics)
    # print("Test analysis added.")
    # retrieved = get_analysis_by_filename(db, "test_file.mp3")
    # if retrieved:
    #     print(f"Retrieved: {retrieved.original_filename}, Timestamp: {retrieved.analysis_timestamp}")
    #     print(recreate_metrics_from_db(retrieved))
    # db.close()
