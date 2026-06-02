from sqlmodel import SQLModel, create_engine, Session

engine = create_engine("sqlite:///horus.db")

def crear_tablas():
    SQLModel.metadata.create_all(engine)

def get_session():
    with Session(engine) as session:
        yield session