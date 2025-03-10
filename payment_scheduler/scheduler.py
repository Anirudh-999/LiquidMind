import pandas as pd
import numpy as np
from datetime import datetime, timedelta

np.random.seed(42)  

invoice_types = ['electric_bill', 'staff_payment', 'raw_materials', 'rent', 'miscellaneous']
importance_weights = {
    'loan': 5,
    'electric_bill': 3,
    'staff_payment': 3,
    'raw_materials': 2,
    'rent': 2,
    'miscellaneous': 1
}

# Learning parameters
alpha = 0.1  # Adjustment for delay-based learning
beta = 0.05  # Adjustment for early payments
gamma = 0.02  # Adjustment for user-prioritized payments
decay_factor = 0.99  # Gradual decay to stabilize weights

#-----------------------------------------------------------------------------------------------
start_date = datetime(2025, 2, 1)
end_date = datetime(2025, 4, 30)
date_range = pd.date_range(start_date, end_date)

payments_data = []
for i in range(1000):
    invoice_type = np.random.choice(invoice_types)
    amount = np.random.randint(100, 2000) 
    due_date = np.random.choice(date_range)  
    payments_data.append({'id': i+1, 'invoice_type': invoice_type, 'amount': amount, 'due_date': due_date})

payments_df = pd.DataFrame(payments_data)
payments_df['due_date'] = pd.to_datetime(payments_df['due_date'])

cash_flow_data = []
for date in date_range:
    inflow = np.random.randint(200, 5000)  
    cash_flow_data.append({'date': date, 'inflow': inflow})

cash_flow_df = pd.DataFrame(cash_flow_data)
cash_flow_df['date'] = pd.to_datetime(cash_flow_df['date'])
cash_flow_df.set_index('date', inplace=True)

payments_df.to_csv('generated_payments.csv', index=False)
cash_flow_df.to_csv('generated_cashflows.csv')
#-----------------------------------------------------------------------------------------------

def compute_priority(row, current_date):
    """
    Compute priority score for a payment given the current date.
    A higher score means the payment is more urgent/important.
    """
    weight = importance_weights.get(row['invoice_type'], 1)
    days_until_due = (row['due_date'] - current_date).days
    days_until_due = max(days_until_due, 0)  
    score = weight / (days_until_due + 1)
    return score

def update_weights(payment_log):
    """
    Adjust weights based on payment delays and user priority patterns.
    - Increase weight if payment was delayed.
    - Adjust relative weights if user consistently pays one type before another.
    """
    global importance_weights
    payment_order = {}  # Tracks the order of payments each day

    for entry in payment_log:
        invoice_type = entry['invoice_type']
        due_date = entry['due_date']
        paid_date = entry['scheduled_date']
        days_late = (paid_date - due_date).days

        if invoice_type not in importance_weights:
            continue

        # Adjust for late payments
        if days_late > 2:
            importance_weights[invoice_type] += alpha
        elif days_late < -3:
            importance_weights[invoice_type] -= beta

        # Track order of payments on the same day
        if paid_date not in payment_order:
            payment_order[paid_date] = []
        payment_order[paid_date].append(invoice_type)

    # Adjust weights based on user preferences
    for date, invoices in payment_order.items():
        for i in range(len(invoices)):
            for j in range(i + 1, len(invoices)):
                first_paid = invoices[i]
                later_paid = invoices[j]

                if first_paid in importance_weights and later_paid in importance_weights:
                    importance_weights[first_paid] += gamma
                    importance_weights[later_paid] -= gamma

    # Apply decay factor to avoid runaway importance
    for key in importance_weights:
        importance_weights[key] *= decay_factor
        importance_weights[key] = max(0.5, min(importance_weights[key], 5))  # Ensure reasonable range

def schedule_payments(payments_df, cash_flow_df, initial_cash=1000):
    pending_payments = payments_df.copy()
    pending_payments['scheduled_date'] = pd.NaT

    available_cash = initial_cash
    schedule = []

    start_date = min(cash_flow_df.index.min(), pending_payments['due_date'].min())
    end_date = max(cash_flow_df.index.max(), pending_payments['due_date'].max()) + timedelta(days=5)
    
    current_date = start_date
    while current_date <= end_date:
        if current_date in cash_flow_df.index:
            available_cash += cash_flow_df.loc[current_date, 'inflow']

        pending = pending_payments[pending_payments['scheduled_date'].isna()]
        if not pending.empty:
            pending = pending.copy()
            pending['priority'] = pending.apply(lambda row: compute_priority(row, current_date), axis=1)
            pending = pending.sort_values(by='priority', ascending=False)
            
            daily_payments = []  # Track payments made today

            for idx, row in pending.iterrows():
                if available_cash >= row['amount']:
                    pending_payments.at[idx, 'scheduled_date'] = current_date
                    available_cash -= row['amount']
                    schedule.append({
                        'date': current_date,
                        'payment_id': row['id'],
                        'invoice_type': row['invoice_type'],
                        'amount': row['amount'],
                        'due_date': row['due_date'],
                        'scheduled_date': current_date
                    })
                    daily_payments.append(row['invoice_type'])

        current_date += timedelta(days=1)

    # Update weights based on actual payments
    update_weights(schedule)
    
    return schedule, pending_payments

schedule, scheduled_payments = schedule_payments(payments_df, cash_flow_df)

schedule_df = pd.DataFrame(schedule)
schedule_df.to_csv('payment_schedule.csv', index=False)

# Print updated weights for review
print("Updated Importance Weights:", importance_weights)
