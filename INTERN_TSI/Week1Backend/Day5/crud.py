from Day1_week1Backend.Day5.models import Employee


def get_employees(db):
    return db.query(Employee).order_by(Employee).all()


def create_employee(db, employee_data):
    employee = Employee(
        name=employee_data.name,
        department=employee_data.department,
        salary=employee_data.salary
    )

    db.add(employee)
    db.commit()
    db.refresh(employee)
    return employee


def get_employee_by_id(db, employee_id):
    return (
        db.query(Employee)
        .filter(Employee.id == employee_id)
        .first()
    )


def update_employee(db, employee_id, employee_data):
    employee = (
        db.query(Employee)
        .filter(Employee.id == employee_id)
        .first()
    )

    if employee is None:
        return None

    employee.name = employee_data.name
    employee.department = employee_data.department
    employee.salary = employee_data.salary

    db.commit()
    db.refresh(employee)
    return employee


def delete_employee(db, employee_id):
    employee = (
        db.query(Employee)
        .filter(Employee.id == employee_id)
        .first()
    )

    if employee is None:
        return None

    db.delete(employee)
    db.commit()

    return employee