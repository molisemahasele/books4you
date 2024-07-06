from flask import Blueprint, render_template, request, flash, jsonify, redirect, url_for, Response
from flask_login import login_required, current_user
from .models import Note
from .models import Event
from .models import Income, Expense, Assets, Liabilities
from . import db
import json
import pandas as pd
import plotly
import plotly.express as px
from sqlalchemy.sql import func
import plotly.graph_objs as go
import numpy as np  # Import NumPy for statistical calculations
from sklearn.linear_model import LogisticRegression  # Import Logistic Regression model
from sklearn.svm import SVR  # Import Support Vector Regressor
from sklearn.tree import DecisionTreeRegressor  # Import Decision Tree Regressor
from sklearn.linear_model import LinearRegression
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from sklearn.preprocessing import MinMaxScaler 
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from collections import Counter
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch
from io import BytesIO

views = Blueprint('views', __name__)

from datetime import datetime, timedelta

from collections import defaultdict, Counter

@views.route('/generate_balance_sheet_pdf', methods=['GET'])
@login_required
def generate_balance_sheet_pdf():
    # Fetch assets and liabilities for the current user
    assets = Assets.query.filter_by(user_id=current_user.id).all()
    liabilities = Liabilities.query.filter_by(user_id=current_user.id).all()

    # Calculate totals
    total_assets_value = sum(asset.price for asset in assets)
    total_depreciation = sum(asset.price * (asset.depreciation / 100.0) for asset in assets)
    net_assets_value = total_assets_value - total_depreciation
    total_liabilities = sum(liability.price for liability in liabilities)
    net_worth = net_assets_value - total_liabilities

    # Create a PDF file in memory
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    line_height = 14

    # Title
    c.setFont("Helvetica-Bold", 16)
    c.drawString(72, height - 36, "Balance Sheet")

    # Draw table
    c.setFont("Helvetica", 12)
    y = height - 72

    def draw_row(headers, columns):
        c.drawString(72, y, headers[0])
        c.drawString(240, y, headers[1])
        c.drawString(420, y, headers[2])
        for col in columns:
            c.drawString(72, col[0], col[1])
            c.drawString(240, col[2])
            c.drawString(420, col[3])
            col[0] -= line_height

    # Headers
    draw_row(["Description", "Original Value (LSL)", "Net Value (LSL)"], [])

    # Assets
    y -= 24
    c.drawString(72, y, "Assets")
    y -= line_height
    for asset in assets:
        depreciation_value = asset.price * (asset.depreciation / 100.0)
        net_value = asset.price - depreciation_value
        c.drawString(72, y, asset.description)
        c.drawString(240, y, f"{asset.price:.2f}")
        c.drawString(420, y, f"{net_value:.2f}")
        y -= line_height

    c.drawString(72, y, "Total Original Assets")
    c.drawString(240, y, f"{total_assets_value:.2f}")
    y -= line_height
    c.drawString(72, y, "Total Depreciation")
    c.drawString(240, y, f"{total_depreciation:.2f}")
    y -= line_height
    c.drawString(72, y, "Total Net Assets")
    c.drawString(240, y, f"{net_assets_value:.2f}")
    y -= line_height * 2

    # Liabilities
    c.drawString(72, y, "Liabilities")
    y -= line_height
    for liability in liabilities:
        c.drawString(72, y, liability.description)
        c.drawString(240, y, f"{liability.price:.2f}")
        y -= line_height

    c.drawString(72, y, "Total Liabilities")
    c.drawString(240, y, f"{total_liabilities:.2f}")
    y -= line_height * 2

    # Net Worth
    c.drawString(72, y, "Net Worth")
    c.drawString(240, y, f"{net_worth:.2f}")

    # Save PDF to buffer
    c.showPage()
    c.save()
    buffer.seek(0)

    return Response(buffer, mimetype='application/pdf',
                    headers={"Content-Disposition": "attachment;filename=balance_sheet.pdf"})


@views.route('/generate_income_statement_pdf', methods=['GET'])
@login_required
def generate_income_statement_pdf():
    # Fetch income, expense, and asset records for the current user
    incomes = Income.query.filter_by(user_id=current_user.id).all()
    expenses = Expense.query.filter_by(user_id=current_user.id).all()
    assets = Assets.query.filter_by(user_id=current_user.id).all()

    # Calculate totals
    total_income = sum(income.total for income in incomes)
    total_expense = sum(expense.price for expense in expenses)
    total_depreciation = sum(asset.price * (asset.depreciation / 100.0) for asset in assets)
    net_income = total_income - total_expense - total_depreciation

    # Create a PDF file in memory
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    line_height = 14

    # Title
    c.setFont("Helvetica-Bold", 16)
    c.drawString(72, height - 36, "Income Statement")

    # Draw table
    c.setFont("Helvetica", 12)
    y = height - 72

    def draw_row(headers, columns):
        c.drawString(72, y, headers[0])
        c.drawString(240, y, headers[1])
        c.drawString(420, y, headers[2])
        for col in columns:
            c.drawString(72, col[0], col[1])
            c.drawString(240, col[2])
            c.drawString(420, col[3])
            col[0] -= line_height

    # Headers
    draw_row(["Product/Description", "Amount (LSL)", "Timestamp"], [])

    # Incomes
    y -= 24
    c.drawString(72, y, "Income")
    y -= line_height
    for income in incomes:
        c.drawString(72, y, income.product)
        c.drawString(240, y, f"{income.total:.2f}")
        c.drawString(420, y, income.timestamp.strftime('%Y-%m-%d'))
        y -= line_height

    c.drawString(72, y, "Total Income")
    c.drawString(240, y, f"{total_income:.2f}")
    y -= line_height * 2

    # Expenses
    c.drawString(72, y, "Expenses")
    y -= line_height
    for expense in expenses:
        c.drawString(72, y, expense.description)
        c.drawString(240, y, f"{expense.price:.2f}")
        c.drawString(420, y, expense.timestamp.strftime('%Y-%m-%d'))
        y -= line_height

    c.drawString(72, y, "Total Expenses")
    c.drawString(240, y, f"{total_expense:.2f}")
    y -= line_height * 2

    # Depreciation
    c.drawString(72, y, "Depreciation")
    y -= line_height
    for asset in assets:
        depreciation_value = asset.price * (asset.depreciation / 100.0)
        c.drawString(72, y, asset.description)
        c.drawString(240, y, f"{depreciation_value:.2f}")
        c.drawString(420, y, asset.timestamp.strftime('%Y-%m-%d'))
        y -= line_height

    c.drawString(72, y, "Total Depreciation")
    c.drawString(240, y, f"{total_depreciation:.2f}")
    y -= line_height * 2

    # Net Income
    c.drawString(72, y, "Net Income")
    c.drawString(240, y, f"{net_income:.2f}")

    # Save PDF to buffer
    c.showPage()
    c.save()
    buffer.seek(0)

    return Response(buffer, mimetype='application/pdf',
                    headers={"Content-Disposition": "attachment;filename=income_statement.pdf"})


@views.route('/generate_audit_report_pdf', methods=['GET'])
@login_required
def generate_audit_report_pdf():
    # Fetch relevant data
    assets = Assets.query.filter_by(user_id=current_user.id).all()
    liabilities = Liabilities.query.filter_by(user_id=current_user.id).all()
    incomes = Income.query.filter_by(user_id=current_user.id).all()
    expenses = Expense.query.filter_by(user_id=current_user.id).all()

    # Create a PDF file in memory
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    line_height = 14

    # Title
    c.setFont("Helvetica-Bold", 16)
    c.drawString(72, height - 36, "Audit Report")

    # Draw content
    c.setFont("Helvetica", 12)
    y = height - 72

    # General Overview
    c.drawString(72, y, "Overview of Financial Data")
    y -= line_height
    c.drawString(72, y, f"Total Assets: {sum(asset.price for asset in assets):.2f}")
    y -= line_height
    c.drawString(72, y, f"Total Liabilities: {sum(liability.price for liability in liabilities):.2f}")
    y -= line_height
    c.drawString(72, y, f"Total Income: {sum(income.total for income in incomes):.2f}")
    y -= line_height
    c.drawString(72, y, f"Total Expenses: {sum(expense.price for expense in expenses):.2f}")

    # Additional details would go here...

    # Save PDF to buffer
    c.showPage()
    c.save()
    buffer.seek(0)

    return Response(buffer, mimetype='application/pdf',
                    headers={"Content-Disposition": "attachment;filename=audit_report.pdf"})


@views.route('/generate_cashflow_statement_pdf', methods=['GET'])
@login_required
def generate_cashflow_statement_pdf():
    # Fetch records for the current user
    incomes = Income.query.filter_by(user_id=current_user.id).all()
    expenses = Expense.query.filter_by(user_id=current_user.id).all()
    assets = Assets.query.filter_by(user_id=current_user.id).all()

    # Calculate cash flows
    operating_activities = sum(income.total for income in incomes) - sum(expense.price for expense in expenses)
    investing_activities = -sum(asset.price for asset in assets if asset.classification == "Fixed Asset")
    financing_activities = 0  # Assuming no data for financing activities
    net_cash_flow = operating_activities + investing_activities + financing_activities

    # Create a PDF file in memory
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    line_height = 14

    # Title
    c.setFont("Helvetica-Bold", 16)
    c.drawString(72, height - 36, "Cash Flow Statement")

    # Draw table
    c.setFont("Helvetica", 12)
    y = height - 72

    # Cash Flow Sections
    c.drawString(72, y, "Cash Flows from Operating Activities")
    y -= line_height
    c.drawString(72, y, f"Operating Activities: {operating_activities:.2f}")
    y -= line_height * 2

    c.drawString(72, y, "Cash Flows from Investing Activities")
    y -= line_height
    c.drawString(72, y, f"Investing Activities: {investing_activities:.2f}")
    y -= line_height * 2

    c.drawString(72, y, "Cash Flows from Financing Activities")
    y -= line_height
    c.drawString(72, y, f"Financing Activities: {financing_activities:.2f}")
    y -= line_height * 2

    c.drawString(72, y, "Net Cash Flow")
    c.drawString(240, y, f"{net_cash_flow:.2f}")

    # Save PDF to buffer
    c.showPage()
    c.save()
    buffer.seek(0)

    return Response(buffer, mimetype='application/pdf',
                    headers={"Content-Disposition": "attachment;filename=cashflow_statement.pdf"})



@views.route('/generate_retained_earnings_statement_pdf', methods=['GET'])
@login_required
def generate_retained_earnings_statement_pdf():
    # Example retained earnings data
    retained_earnings_beginning = 5000.00  # Replace with actual data
    net_income = sum(income.total for income in Income.query.filter_by(user_id=current_user.id).all())
    dividends_paid = 0  # Replace with actual data if available
    retained_earnings_end = retained_earnings_beginning + net_income - dividends_paid

    # Create a PDF file in memory
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    line_height = 14

    # Title
    c.setFont("Helvetica-Bold", 16)
    c.drawString(72, height - 36, "Statement of Retained Earnings")

    # Draw table
    c.setFont("Helvetica", 12)
    y = height - 72

    c.drawString(72, y, "Retained Earnings (Beginning)")
    c.drawString(240, y, f"{retained_earnings_beginning:.2f}")
    y -= line_height

    c.drawString(72, y, "Net Income")
    c.drawString(240, y, f"{net_income:.2f}")
    y -= line_height

    c.drawString(72, y, "Dividends Paid")
    c.drawString(240, y, f"{dividends_paid:.2f}")
    y -= line_height

    c.drawString(72, y, "Retained Earnings (End)")
    c.drawString(240, y, f"{retained_earnings_end:.2f}")

    # Save PDF to buffer
    c.showPage()
    c.save()
    buffer.seek(0)

    return Response(buffer, mimetype='application/pdf',
                    headers={"Content-Disposition": "attachment;filename=retained_earnings_statement.pdf"})


@views.route('/generate_trial_balance_pdf', methods=['GET'])
@login_required
def generate_trial_balance_pdf():
    # Example trial balance data
    # Fetch all entries and categorize them
    incomes = Income.query.filter_by(user_id=current_user.id).all()
    expenses = Expense.query.filter_by(user_id=current_user.id).all()
    assets = Assets.query.filter_by(user_id=current_user.id).all()
    liabilities = Liabilities.query.filter_by(user_id=current_user.id).all()

    # Calculate totals
    total_incomes = sum(income.total for income in incomes)
    total_expenses = sum(expense.price for expense in expenses)
    total_assets = sum(asset.price for asset in assets)
    total_liabilities = sum(liability.price for liability in liabilities)

    # Create a PDF file in memory
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    line_height = 14

    # Title
    c.setFont("Helvetica-Bold", 16)
    c.drawString(72, height - 36, "Trial Balance")

    # Draw table
    c.setFont("Helvetica", 12)
    y = height - 72

    # Draw trial balance headers and entries
    c.drawString(72, y, "Description")
    c.drawString(240, y, "Debit (LSL)")
    c.drawString(420, y, "Credit (LSL)")
    y -= line_height * 2

    for income in incomes:
        c.drawString(72, y, income.product)
        c.drawString(240, y, f"{income.total:.2f}")
        y -= line_height

    for expense in expenses:
        c.drawString(72, y, expense.description)
        c.drawString(420, y, f"{expense.price:.2f}")
        y -= line_height

    for asset in assets:
        c.drawString(72, y, asset.description)
        c.drawString(240, y, f"{asset.price:.2f}")
        y -= line_height

    for liability in liabilities:
        c.drawString(72, y, liability.description)
        c.drawString(420, y, f"{liability.price:.2f}")
        y -= line_height

    # Total row
    c.drawString(72, y, "Total")
    c.drawString(240, y, f"{total_assets + total_incomes:.2f}")
    c.drawString(420, y, f"{total_liabilities + total_expenses:.2f}")

    # Save PDF to buffer
    c.showPage()
    c.save()
    buffer.seek(0)

    return Response(buffer, mimetype='application/pdf',
                    headers={"Content-Disposition": "attachment;filename=trial_balance.pdf"})



@views.route('/statistics')
@login_required
def statistics():
    # Get total incomes and total expenses for the current user
    total_incomes = db.session.query(func.sum(Income.total)).filter_by(user_id=current_user.id).scalar() or 0
    total_expenses = db.session.query(func.sum(Expense.price)).filter_by(user_id=current_user.id).scalar() or 0
    
    # Calculate asset turnover ratio (Revenue / Total Assets)
    total_assets = db.session.query(func.sum(Assets.price)).filter_by(user_id=current_user.id).scalar() or 1  # Prevent division by zero
    asset_turnover_ratio = total_incomes / total_assets
    
    # Calculate debt-to-equity ratio (Total Liabilities / Total Assets)
    total_liabilities = db.session.query(func.sum(Liabilities.price)).filter_by(user_id=current_user.id).scalar() or 0
    debt_to_equity_ratio = total_liabilities / total_assets if total_assets != 0 else 0
    
    # Calculate profit margin (Net Income / Revenue)
    profit_margin = ((total_incomes - total_expenses) / total_incomes * 100) if total_incomes != 0 else 0
    
    # Example calculation for revenue growth
    # You need historical data to calculate actual growth rates. This is a placeholder.
    previous_period_revenue = total_incomes * 0.9  # Example previous period revenue
    revenue_growth = ((total_incomes - previous_period_revenue) / previous_period_revenue * 100) if previous_period_revenue != 0 else 0

    # Get the frequency of each description in expenses for the current user
    expense_descriptions = db.session.query(Expense.description).filter_by(user_id=current_user.id).all()
    description_counts = Counter([desc[0] for desc in expense_descriptions])

    # Get expense data for the line chart
    expenses = db.session.query(Expense.timestamp, Expense.price).filter_by(user_id=current_user.id).all()
    expense_data = defaultdict(list)
    for timestamp, price in expenses:
        expense_data['date'].append(timestamp.strftime('%Y-%m-%d'))
        expense_data['price'].append(price)

    # Get income data for the line chart
    incomes = db.session.query(Income.timestamp, Income.total).filter_by(user_id=current_user.id).all()
    income_data = defaultdict(list)
    for timestamp, total in incomes:
        income_data['date'].append(timestamp.strftime('%Y-%m-%d'))
        income_data['total'].append(total)

    data = {
        'total_incomes': total_incomes,
        'total_expenses': total_expenses,
        'total_liabilities': total_liabilities,
        'total_assets': total_assets,
        'description_counts': description_counts,
        'expense_data': expense_data,
        'income_data': income_data,
        'asset_turnover_ratio': asset_turnover_ratio,
        'debt_to_equity_ratio': debt_to_equity_ratio,
        'revenue_growth': revenue_growth,
        'profit_margin': profit_margin
    }

    return render_template('statistics.html', data=data)




@views.route('/', methods=['GET', 'POST'])
def landing_page():
    return render_template('landing_page.html')

@views.route('/home', methods=['GET', 'POST'])
@login_required
def home():
    if request.method == 'POST':
        # Retrieve form data
        date_str = request.form.get('date')
        quantity = request.form.get('qty')
        product = request.form.get('product')
        price = request.form.get('price')
        note = request.form.get('note')

        # Validate and convert data
        try:
            timestamp = datetime.strptime(date_str, '%Y-%m-%d').date()
            quantity = int(quantity)
            price = int(price)
            total = quantity * price  # Calculate total
        except (ValueError, TypeError) as e:
            flash("Invalid input data. Please check your entries.")
            return redirect(url_for('views.home'))

        # Create new Income record
        new_Income = Income(
            timestamp=timestamp,
            quantity=quantity,
            product=product,
            price=price,
            total=total,
            note=note,
            user_id=current_user.id  # Replace with actual user_id, possibly from session or context
        )

        # Add and commit to the database
        try:
            db.session.add(new_Income)
            db.session.commit()
            flash("Record added successfully!")
        except Exception as e:
            db.session.rollback()
            flash("Error adding record to the database. Please try again.")


        return redirect(url_for('views.home'))
    
    records = Income.query.filter_by(user_id=current_user.id).all()
    
    # Handle GET request
    return render_template('index.html', records=records)


@views.route('/expenses', methods=['GET', 'POST'])
@login_required
def expenses():
    if request.method == 'POST':
        # Retrieve form data
        date_str = request.form.get('date')
        description = request.form.get('description')
        price = request.form.get('price')
        supplier = request.form.get('supplier')
        method = request.form.get('method')
        invoice_no = request.form.get('invoice_no')

        # Validate and convert data
        try:
            timestamp = datetime.strptime(date_str, '%Y-%m-%d').date()
            price = int(price)
        except (ValueError, TypeError) as e:
            flash("Invalid input data. Please check your entries.")
            return redirect(url_for('views.expenses'))

        # Create new Expense record
        new_expense = Expense(
            timestamp=timestamp,
            description=description,
            price=price,
            supplier=supplier,
            method=method,
            invoice_no=invoice_no,
            user_id=current_user.id  # Use current_user.id
        )

        # Add and commit to the database
        try:
            db.session.add(new_expense)
            db.session.commit()
            flash("Expense added successfully!")
        except Exception as e:
            db.session.rollback()
            flash("Error adding expense to the database. Please try again.")
            print(e)

        return redirect(url_for('views.expenses'))
    
    # Retrieve expenses for the current user only
    expenses = Expense.query.filter_by(user_id=current_user.id).all()

    # Handle GET request
    return render_template('expenses.html', expenses=expenses)


@views.route('/assets', methods=['GET', 'POST'])
@login_required
def assets():
    if request.method == 'POST':
        # Retrieve form data
        date_str = request.form.get('date')
        description = request.form.get('description')
        price = request.form.get('price')
        supplier = request.form.get('supplier')
        classification = request.form.get('classification')
        depreciation = request.form.get('depreciation')

        # Validate and convert data
        try:
            timestamp = datetime.strptime(date_str, '%Y-%m-%d').date()
            price = float(price)
            depreciation = int(depreciation)
        except (ValueError, TypeError) as e:
            flash("Invalid input data. Please check your entries.")
            return redirect(url_for('views.assets'))

        # Create new Assets record
        new_asset = Assets(
            timestamp=timestamp,
            description=description,
            price=price,
            supplier=supplier,
            classification=classification,
            depreciation=depreciation,
            user_id=current_user.id
        )

        # Add and commit to the database
        try:
            db.session.add(new_asset)
            db.session.commit()
            flash("Asset added successfully!")
        except Exception as e:
            db.session.rollback()
            flash("Error adding asset to the database. Please try again.")

        return redirect(url_for('views.assets'))

    # Retrieve assets for the current user
    assets = Assets.query.filter_by(user_id=current_user.id).all()

    return render_template('assets.html', assets=assets)

@views.route('/liabilities', methods=['GET', 'POST'])
@login_required
def liabilities():
    if request.method == 'POST':
        # Retrieve form data
        date_str = request.form.get('date')
        description = request.form.get('description')
        price = request.form.get('price')
        lender = request.form.get('supplier')  # Using 'supplier' input for lender
        classification = request.form.get('classification')

        # Validate and convert data
        try:
            timestamp = datetime.strptime(date_str, '%Y-%m-%d').date()
            price = int(price)
        except (ValueError, TypeError) as e:
            flash("Invalid input data. Please check your entries.")
            return redirect(url_for('views.liabilities'))

        # Create new Liabilities record
        new_liability = Liabilities(
            timestamp=timestamp,
            description=description,
            price=price,
            lender=lender,
            classification=classification,
            user_id=current_user.id
        )

        # Add and commit to the database
        try:
            db.session.add(new_liability)
            db.session.commit()
            flash("Liability added successfully!")
        except Exception as e:
            db.session.rollback()
            flash("Error adding liability to the database. Please try again.")

        return redirect(url_for('views.liabilities'))
    
    # Handle GET request
    liabilities = Liabilities.query.filter_by(user_id=current_user.id).all()
    return render_template('liabilities.html', liabilities=liabilities)

@views.route('/delete_liability/<int:liability_id>', methods=['DELETE'])
@login_required
def delete_liability(liability_id):
    # Find the liability by id
    liability = Liabilities.query.get_or_404(liability_id)
    
    # Ensure the liability belongs to the current user
    if liability.user_id != current_user.id:
        return jsonify({'error': 'Unauthorized access.'}), 403
    
    try:
        # Delete the liability
        db.session.delete(liability)
        db.session.commit()
        return jsonify({'success': True}), 200
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500


@views.route('/delete/<int:record_id>', methods=['DELETE'])
def delete_record(record_id):
    # Find the record by id
    record = Income.query.get_or_404(record_id)
    
    try:
        # Delete the record
        db.session.delete(record)
        db.session.commit()
        return jsonify({'success': True}), 200
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500
    
@views.route('/delete_expense/<int:expense_id>', methods=['DELETE'])
@login_required
def delete_expense(expense_id):
    # Find the expense by id
    expense = Expense.query.get_or_404(expense_id)
    
    if expense.user_id != current_user.id:
        return jsonify({'error': 'Unauthorized access.'}), 403
    
    try:
        # Delete the expense
        db.session.delete(expense)
        db.session.commit()
        return jsonify({'success': True}), 200
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500
    

@views.route('/delete_asset/<int:asset_id>', methods=['DELETE'])
@login_required
def delete_asset(asset_id):
    # Find the asset by id
    asset = Assets.query.get_or_404(asset_id)
    
    # Ensure the asset belongs to the current user
    if asset.user_id != current_user.id:
        return jsonify({'error': 'Unauthorized access.'}), 403
    
    try:
        # Delete the asset
        db.session.delete(asset)
        db.session.commit()
        return jsonify({'success': True}), 200
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500
    
@views.route('/edit_liability/<int:liability_id>', methods=['GET', 'POST'])
@login_required
def edit_liability(liability_id):
    liability = Liabilities.query.get_or_404(liability_id)
    
    if liability.user_id != current_user.id:
        flash('Unauthorized access.')
        return redirect(url_for('views.liabilities'))
    
    if request.method == 'POST':
        date_str = request.form.get('date')
        description = request.form.get('description')
        price = request.form.get('price')
        lender = request.form.get('supplier')  # Using 'supplier' input for lender
        classification = request.form.get('classification')

        try:
            timestamp = datetime.strptime(date_str, '%Y-%m-%d').date()
            price = int(price)
        except (ValueError, TypeError) as e:
            flash("Invalid input data. Please check your entries.")
            return redirect(url_for('views.liabilities'))

        liability.timestamp = timestamp
        liability.description = description
        liability.price = price
        liability.lender = lender
        liability.classification = classification

        try:
            db.session.commit()
            flash("Liability updated successfully!")
        except Exception as e:
            db.session.rollback()
            flash("Error updating liability. Please try again.")
        
        return redirect(url_for('views.liabilities'))
    
    return jsonify({
        'timestamp': liability.timestamp.strftime('%Y-%m-%d'),
        'description': liability.description,
        'price': liability.price,
        'lender': liability.lender,
        'classification': liability.classification
    })





"""@views.route('/update/<int:id>', methods=['GET', 'POST'])
def update(id):
    note_to_update = Note.query.get_or_404(id)
    if request.method == "POST":
        note_to_update.data = request.form['data']
        try:
            db.session.commit()
            return redirect('/')
        except:
            return "There was an error"
    return render_template("update.html", note_to_update=note_to_update, user=current_user)

"""


"""@views.route('/delete-note', methods=['POST'])
def delete_note():
    note = json.loads(request.data)
    noteId = note['noteId']
    note = Note.query.get(noteId)
    if note:
        if note.user_id == current_user.id:
            db.session.delete(note)
            db.session.commit()
            
    return jsonify({})"""