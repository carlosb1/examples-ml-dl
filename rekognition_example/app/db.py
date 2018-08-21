from sqlalchemy import Column, ForeignKey, Integer, String, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy import create_engine

Base = declarative_base()


class File(Base):
    __tablename__ = 'file'
    id = Column(Integer, primary_key=True)
    filepath = Column(String(250), nullable=False)


class Label(Base):
    __tablename__ = 'label'
    id = Column(Integer, primary_key=True)
    confidence = Column(Float)
    name = Column(String(250), nullable=False)
    file_id = Column(Integer, ForeignKey('file.id'))
    fil = relationship(File)


engine = create_engine('sqlite:///database.db')
Base.metadata.create_all(engine)
