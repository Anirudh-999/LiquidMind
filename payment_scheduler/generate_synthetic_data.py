import pandas as pd
import random
from datetime import datetime, timedelta

# Parameters for generating sample data
num_invoices = 100
start_date = datetime.now()
due_dates = [start_date + timedelta(days=random.randint(1, 30)) for _ in range(num_invoices)]
amounts = [random.randint(100, 10000) for _ in range(num_invoices)]
interest_rates = [random.uniform(0.01, 0.2) for _ in range(num_invoices)]
paid_on_time = [random.choice([0, 1]) for _ in range(num_invoices)]  # 0 = not paid, 1 = paid

# Create a DataFrame
data = {
    'invoice_id': [f'INV{str(i).zfill(4)}' for i in range(num_invoices)],
    'due_date': due_dates,
    'amount': amounts,
    'interest_rate': interest_rates,
    'paid_on_time': paid_on_time
}

df = pd.DataFrame(data)

# Save to CSV
df.to_csv('invoices.csv', index=False)
print("invoices.csv has been generated.")
