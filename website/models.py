from . import db
from flask_login import UserMixin
from sqlalchemy.sql import func

class Note(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    data = db.Column(db.String(150))
    date = db.Column(db.DateTime(timezone=True), default=func.now())
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))

class Event(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime(timezone=True))
    event_type = db.Column(db.String(50))  # Assuming event type has a limited range of options
    reason = db.Column(db.String(255))  # Reason might be lengthy
    operator = db.Column(db.String(50))  # Assuming operator has a limited range of options
    capacity = db.Column(db.Integer)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))

class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(150), unique=True)
    password = db.Column(db.String(150))
    first_name = db.Column(db.String(150))
    notes = db.relationship('Note')
    events = db.relationship('Event', backref='user')
    

class Income(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.Date)  # Changed to Date
    quantity = db.Column(db.Integer) 
    product = db.Column(db.String(255))  
    price = db.Column(db.Integer)  
    total = db.Column(db.Integer)
    note = db.Column(db.String(225))
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))

class Expense(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.Date)  # Changed to Date 
    description = db.Column(db.String(255))  
    price = db.Column(db.Integer)  
    supplier = db.Column(db.String(225))
    method = db.Column(db.String(225))
    invoice_no = db.Column(db.String(225))
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))

class Assets(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.Date)  # Changed to Date 
    description = db.Column(db.String(255))  
    price = db.Column(db.Integer)  
    supplier = db.Column(db.String(225))
    classification = db.Column(db.String(225))
    depreciation = db.Column(db.Integer)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))

class Liabilities(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.Date)  # Changed to Date 
    description = db.Column(db.String(255))  
    price = db.Column(db.Integer)  
    lender = db.Column(db.String(225))
    classification = db.Column(db.String(225))
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))


    
    # Additional attributes if needed, like duration for breakdowns etc.

