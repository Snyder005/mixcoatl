"""Crosstalk database classes and functions.

To Do:
    * Expand methods for database querying and retrieval of objects.
"""
import sqlalchemy as sql
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, backref, sessionmaker, aliased
from contextlib import contextmanager
from sqlalchemy.orm.exc import NoResultFound
from sqlalchemy.orm.collections import attribute_mapped_collection

from mixcoatl.errors import AlreadyExists, MissingKeyword

Base = declarative_base()
Session = sessionmaker()

@contextmanager
def db_session(database, echo=False):
    """Define a context manager for a database session.
    
    Parameters
    ----------
    database : `str` 
        Filepath to SQLite database.
    echo : `bool`
        `True` to enable Engine to log all statements to log handler, which
        defaults to `sys.stdout`.
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
    aggressor_id = sql.Column(sql.Integer, sql.ForeignKey('segment.id'), 
                              comment='ID for aggressor segment.')
    victim_id = sql.Column(sql.Integer, sql.ForeignKey('segment.id'), 
                           comment='ID for victim segment.')
    aggressor_signal = sql.Column(sql.Float, comment='Pixel signal of aggressor.')
    coefficient = sql.Column(sql.Float, comment='Crosstalk coefficient.')
    error = sql.Column(sql.Float, comment='Error estimate for crosstalk coefficient.')
    methodology = sql.Column(sql.String, comment='Measurement methodology.')
    image_type = sql.Column(sql.String, comment='Type of image (e.g. satellite).')
    teststand = sql.Column(sql.String, comment='Laboratory test stand.')
    analysis = sql.Column(sql.String, comment='Analysis task (e.g. CrosstalkSatelliteTask).')
    is_coadd = sql.Column(sql.Boolean, comment='Indicator for coadded image.')

    ## Relationships
    aggressor = relationship("Segment", back_populates="results", foreign_keys=[aggressor_id])
    victim = relationship("Segment", foreign_keys=[victim_id])
    
    def __repr__(self):
        return "<Result(aggressor_signal={0:d}, coefficient={1:0.1e}, method='{2}')>".\
            format(self.aggressor_signal, self.coefficient, self.method)

    def add_to_db(self, session):
        """Add Result to database."""
        session.add(self)
        
class Segment(Base):
    
    __tablename__ = 'segment'
    
    ## Columns
    id = sql.Column(sql.Integer, primary_key=True)
    segment_name = sql.Column(sql.String, comment='Segment name (e.g. C00).')
    amplifier_number = sql.Column(sql.Integer, comment='Segment amplifier number.')
    sensor_id = sql.Column(sql.Integer, sql.ForeignKey('sensor.id'), comment='ID for CCD sensor.')
    
    ## Relationships
    results = relationship("Result", back_populates="aggressor", cascade="all, delete-orphan",
                           foreign_keys=[Result.aggressor_id])
    sensor = relationship("Sensor", back_populates="segments")
    
    def __repr__(self):
        return "<Segment(segment_name='{0}', amplifier_number={1})>".format(self.sensor_name,
                                                                            self.amplifier_number)

    @classmethod
    def from_db(cls, session, **kwargs):
        """Initialize Segment from database using a query."""
        query = session.query(cls)

        ## Query on amplifier number or name
        if 'amplifier_number' in kwargs:
            query = query.filter(cls.amplifier_number == kwargs['amplifier_number'])
        elif 'segment_name' in kwargs:
            query = query.filter(cls.segment_name == kwargs['segment_name'])
        else:
            raise MissingKeyword("Missing 'amplifier_number' or 'segment_name' keyword for query.")

	    ## Filter on sensor name
        if 'sensor_name' in kwargs:
            query = query.join(Sensor).filter(Sensor.sensor_name == kwargs['sensor_name'])
        elif 'lsst_num' in kwargs:
            query = query.join(Sensor).filter(Sensor.lsst_num == kwargs['lsst_num'])
        else:
            raise MissingKeyword("Missing 'sensor_name' or 'lsst_num' keyword for query.")

        return query.one()

    def add_to_db(self, session):
        """Add Segment to database."""
        session.add(self)

class Sensor(Base):
    
    __tablename__ = 'sensor'
    
    ## Columns
    id = sql.Column(sql.Integer, primary_key=True)
    sensor_name = sql.Column(sql.String, comment='Sensor name (e.g. R22/S22).')
    lsst_num = sql.Column(sql.String, comment='LSST project number.')
    manufacturer = sql.Column(sql.String, comment='Manufacturer (E2V or ITL).')
    namps = sql.Column(sql.Integer, comment='Number of amplifiers.')
    
    ## Relationships
    segments = relationship("Segment", collection_class=attribute_mapped_collection('amplifier_number'), 
                            cascade="all, delete-orphan", back_populates="sensor")

    def __repr__(self):
        return "<Sensor(sensor_name='{0}', lsst_num='{1}', manufacturer='{2}')>".format(self.sensor_name,
                                                                                        self.lsst_num,
                                                                                        self.manufacturer)
    
    @classmethod
    def from_db(cls, session, **kwargs):
        """Initialize Sensor from database using a query."""
        query = session.query(cls)

        ## Query on name or lsst number
        if 'sensor_name' in kwargs:
            query = query.filter(cls.sensor_name == kwargs['sensor_name'])
        elif 'lsst_num' in kwargs:
            query = query.filter(cls.lsst_num == kwargs['lsst_num'])
        else:
            raise MissingKeyword("Missing 'sensor_name' or 'lsst_num' keyword for query.")
        
        return query.one()
            
    def add_to_db(self, session):
        """Add Sensor to database."""
        session.add(self)

def query_results(session, sensor_name, aggressor_amp, victim_amp, methodology=None):
    """Query database for results."""
    a1 = aliased(Segment)
    a2 = aliased(Segment)
    
    query = session.query(Result).join(a1, Result.aggressor_id == a1.id).\
        join(a2, Result.victim_id == a2.id).\
        filter(a1.amplifier_number == aggressor_amp).\
        filter(a2.amplifier_number == victim_amp).\
        join(Sensor).filter(Sensor.sensor_name == sensor_name)

    if methods is not None:
        query = query.filter(Result.methodology == methodology)

    return query.all()
