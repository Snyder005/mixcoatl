import sqlalchemy as sql
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, backref, sessionmaker, aliased
from contextlib import contextmanager
from sqlalchemy.orm.exc import NoResultFound

from mixcoatl.errors import AlreadyExists, MissingKeyword

Base = declarative_base()
Session = sessionmaker()

@contextmanager
def db_session(database, echo=False):
    """Create a session bound to the given database.
    
    Args:
        database (str): Database filepath.
    """
    try:
        engine = sql.create_engine('sqlite:///{0}'.format(database), echo=echo)
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
    
    @classmethod
    def from_db(cls, session, sensor_name, **kwargs):
        """Initialize Segment from database using a query."""
        query = session.query(cls)

        ## Query on amplifier number or name
        if 'amplifier_number' in kwargs:
            query = query.filter(cls.amplifier_number == kwargs['amplifier_number'])
        elif 'name' in kwargs:
            query = query.filter(cls.name == kwargs['name'])
        else:
            raise MissingKeyword('Missing amplifier_number or name keyword for query.')

	## Filter on sensor name
        query = query.join(Sensor).filter(Sensor.name == sensor_name)

        return query.one()

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
    def from_db(cls, session, sensor_name):
        """Initialize Sensor from database using a query."""
        query = session.query(cls).filter(cls.name == sensor_name)
        
        return query.one()
            
    def add_to_db(self, session):
        """Add Sensor to database."""
        session.add(self)

def query_results(session, sensor_name, aggressor_amp, victim_amp, methods=None):
    """Query database for results."""
    a1 = aliased(Segment)
    a2 = aliased(Segment)
    
    query = session.query(Result).join(a1, Result.aggressor_id == a1.id).\
        join(a2, Result.victim_id == a2.id).\
        filter(a1.amplifier_number == aggressor_amp).\
        filter(a2.amplifier_number == victim_amp).\
        join(Sensor).filter(Sensor.name == sensor_name)

    if methods is not None:
        if not isinstance(methods, list):
            methods = [methods]
        query = query.filter(Result.method.in_(methods))

    return query.all()