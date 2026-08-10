from fastapi import FastAPI, Depends
from sqlalchemy.orm import Session
from .database import SessionLocal
from Day1_week1Backend.Day5.crud import get_employees, create_employee, update_employee, delete_employee 
from Day1_week1Backend.Day4.schemas import EmployeeCreate, EmployeeResponse
from Day1_week1Backend.Day5.crud import get_employee_by_id
from fastapi import HTTPException

app = FastAPI()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@app.get("/")
def test():
    return {"message": "Database connected"}


@app.get("/employees", response_model=list[EmployeeResponse] , tags=["Employees Read"])
def read_employees(
    db: Session = Depends(get_db)
):
    employees = get_employees(db)
    if employees is None:
        http_exception = HTTPException (Status_code=402, detail="Employees not found")
        raise http_exception
    return employees


@app.post("/employees", response_model=EmployeeResponse, tags=["Employees create"])
def add_employee(
    employee: EmployeeCreate,
    db: Session = Depends(get_db)
):
    employees = create_employee(db, employee)
    if employees is None:
        http_exception = HTTPException(Status_code=400, detail="Employee not created")
        raise http_exception
    return employees

@app.get("/employees/{employee_id}", response_model=EmployeeResponse, tags=["Employees Read"])
def read_employee_by_id(
    employee_id: int,
    db: Session = Depends(get_db)
):
    employees = get_employee_by_id(db, employee_id)
    if employees is None:
        http_exception = HTTPException(Status_code=404, detail="Employee not found")
        raise http_exception
    return employees

@app.put("/employees/{employee_id}",response_model=EmployeeResponse, tags=["Employees Update"])
def update_employee_by_id(
    employee_id: int,
    employee: EmployeeCreate,
    db: Session = Depends(get_db)
):
    employees = update_employee(db, employee_id, employee)
    if employees is None:
        http_exception = HTTPException(Status_code= 406, detail="Employee not Updated")
        raise http_exception
    return employees

@app.delete("/employees/{employee_id}",response_model=EmployeeResponse,tags=["Employees Delete"])
def delete_employee_id(
    employee_id: int,
    db: Session = Depends(get_db)
):
    employees = delete_employee(db, employee_id)
    if employees is None:
        http_expception = HTTPException(Status_code=406, detail="Employee is Not Deleted")
        raise http_expception
    return employees