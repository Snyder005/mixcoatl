import sqlalchemy as sql
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, backref, sessionmaker
from contextlib import contextmanager

from mixcoatl.errors import AlreadyExists, MissingKeyword

Base = declarative_base()
Session = sessionmaker()

@contextmanager
def db_session(database):
    """Create a session bound to the given database.
    
    Args:
        database (str): Database filepath.
    """

    try:
        engine = sql.create_engine(database, echo=False)
        Base.metadata.create_all(engine)
        Session.configure(bind=engine)
        session = Session()
        yield session
        session.commit()
    except Exception as e:
        session.rollback()
        raise e
    finally:
        session.close()

class Result(Base):
    
    __tablename__ = 'result'
    
    ## Columns
    id = sql.Column(sql.Integer, primary_key=True)
    aggressor_id = sql.Column(sql.Integer, sql.ForeignKey('segment.id'))
    aggressor_signal = sql.Column(sql.Float)
    coefficient = sql.Column(sql.Float)
    error = sql.Column(sql.Float)
    filename = sql.Column(sql.String)
    method = sql.Column(sql.String)
    victim_id = sql.Column(sql.Integer, sql.ForeignKey('segment.id'))

    ## Relationships
    aggressor = relationship("Segment", back_populates="results", foreign_keys=[aggressor_id])
    victim = relationship("Segment", foreign_keys=[victim_id])
    
    def add_to_db(self, session):
        """Add Result to database."""
        
        session.add(self)
        
class Segment(Base):
    
    __tablename__ = 'segment'
    
    ## Columns
    id = sql.Column(sql.Integer, primary_key=True)
    name = sql.Column(sql.String)
    amplifier_number = sql.Column(sql.Integer)
    sensor_id = sql.Column(sql.Integer, sql.ForeignKey('sensor.id'))
    
    ## Relationships
    results = relationship("Result", back_populates="aggressor", foreign_keys=[Result.aggressor_id])
    sensor = relationship("Sensor", back_populates="segments")
    
    def add_to_db(self, session):
        """Add Segment to database."""
        
        session.add(self)

class Sensor(Base):
    
    __tablename__ = 'sensor'
    
    ## Columns
    id = sql.Column(sql.Integer, primary_key=True)
    name = sql.Column(sql.String)
    manufacturer = sql.Column(sql.String)
    namps = sql.Column(sql.Integer)
    
    ## Relationships
    segments = relationship("Segment", back_populates="sensor")
    
    @classmethod
    def from_db(cls, session, **kwargs):
        
        query = session.query(cls)
        
        ## Query on name or id
        if 'name' in kwargs:
            query = query.filter(cls.name == kwargs['name'])
        elif 'id' in kwargs:
            query = query.filter(cls.id == kwargs['id'])
        else:
            raise MissingKeyword('Query requires name or id keyword.')
            
        sensor = query.one()
        return sensor
            
    def add_to_db(self, session):
        """Add Sensor to database."""
        
        ## Check if already exists in database
        sensor = session.query(Sensor).filter_by(name=self.name).first()
        if sensor is None:
            session.add(self)
        else:
            raise AlreadyExists('Sensor already exists in database.')
