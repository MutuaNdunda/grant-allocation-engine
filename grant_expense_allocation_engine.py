#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 19:30:17 2026

@author: mutua
"""
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from uuid import uuid4
import hashlib
import os

# GRANT ALLOCATION ENGINE CLASS DEFINATION
class GrantAllocationEngine:
    """
    Main engine for allocating expenses to grants based on defined business rules.
    """

    def __init__(self, grants_df: pd.DataFrame, transactions_df: pd.DataFrame):
        """
        Initialize the allocation engine with grants and transactions data.
        """
        self.grants_df = grants_df.copy()
        self.transactions_df = transactions_df.copy()
        self.allocations = []
        self.unallocated = []
        self.allocation_complete = False  # Flag to prevent re-running

        # Pre-process data
        self._prepare_data()


    def _prepare_data(self):
        """Convert date columns and initialize tracking columns."""
        # Convert date columns to datetime
        date_columns = ['StartDate', 'EndDate']
        for col in date_columns:
            if col in self.grants_df.columns:
                self.grants_df[col] = pd.to_datetime(self.grants_df[col])

        if 'TransactionDate' in self.transactions_df.columns:
            self.transactions_df['TransactionDate'] = pd.to_datetime(
                self.transactions_df['TransactionDate']
            )

        # Initialize remaining budget for each grant
        self.grants_df['RemainingBudget'] = self.grants_df['TotalAmount'].copy()

        # Add allocation tracking columns to transactions
        self.transactions_df['AllocatedAmount'] = 0.0
        self.transactions_df['AllocationStatus'] = 'Pending'

    def _is_transaction_eligible_for_grant(self, transaction: pd.Series, grant: pd.Series) -> bool:
        """
        Check if a transaction is eligible for a specific grant based on rules.
        """
        # Check date range
        if not (grant['StartDate'] <= transaction['TransactionDate'] <= grant['EndDate']):
            return False

        # Check remaining budget
        if grant['RemainingBudget'] <= 0:
            return False

        # Define restriction columns to check
        restriction_cols = ['BusinessUnit', 'Country', 'Account', 'ProjectName', 'DepartmentName']

        # Check each restriction column
        for col in restriction_cols:
            if col in grant.index and col in transaction.index:
                grant_value = grant[col]
                trans_value = transaction[col]

                # NULL in grant means wildcard (matches anything)
                if pd.isna(grant_value) or grant_value == '' or grant_value is None:
                    continue

                # If grant has a value, transaction must match exactly
                if pd.isna(trans_value) or grant_value != trans_value:
                    return False

        return True

    def _get_eligible_grants(self, transaction: pd.Series) -> pd.DataFrame:
        """
        Find all grants eligible for a given transaction.
        """
        eligible_mask = self.grants_df.apply(
            lambda grant: self._is_transaction_eligible_for_grant(transaction, grant),
            axis=1
        )

        eligible_grants = self.grants_df[eligible_mask].copy()

        # Sort by Priority ASC (lower number = higher priority), then EndDate ASC
        eligible_grants = eligible_grants.sort_values(
            by=['Priority', 'EndDate'],
            ascending=[True, True]
        )

        return eligible_grants

    def _allocate_amount_to_grant(self, grant: pd.Series, amount: float) -> Tuple[float, float]:
        """
        Allocate a specific amount to a grant, respecting remaining budget.
        """
        if grant['RemainingBudget'] >= amount:
            allocated = amount
            remaining = 0
        else:
            allocated = grant['RemainingBudget']
            remaining = amount - grant['RemainingBudget']

        return allocated, remaining

    def run_allocation(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Execute the complete allocation process.
        Only runs once - subsequent calls return cached results.
        """
        # Return cached results if allocation already done
        if self.allocation_complete:
            allocations_df = pd.DataFrame(self.allocations) if self.allocations else pd.DataFrame()
            unallocated_df = pd.DataFrame(self.unallocated) if self.unallocated else pd.DataFrame()
            burn_rate_df = self._calculate_burn_rate()
            return allocations_df, unallocated_df, burn_rate_df
        
        # Sort transactions by date (earliest first)
        sorted_transactions = self.transactions_df.sort_values('TransactionDate')
        
        # Generate batch ID for idempotency
        batch_id = self._generate_batch_id()
        timestamp = datetime.now()
        
        # Reset allocations and unallocated tracking
        self.allocations = []
        self.unallocated = []
        
        # Reset allocated amount in transactions
        self.transactions_df['AllocatedAmount'] = 0.0
        self.transactions_df['AllocationStatus'] = 'Pending'
        
        for idx, transaction in sorted_transactions.iterrows():
            remaining_amount = transaction['Amount']
            transaction_id = transaction['TransactionId']
            
            # Get eligible grants for this transaction
            eligible_grants = self._get_eligible_grants(transaction)
            
            if eligible_grants.empty:
                # No eligible grants found - entire amount is unallocated
                self.unallocated.append({
                    'TransactionId': transaction_id,
                    'Amount': remaining_amount,
                    'Reason': 'No eligible grants available',
                    'TransactionDate': transaction['TransactionDate']
                })
                self.transactions_df.at[idx, 'AllocatedAmount'] = 0
                self.transactions_df.at[idx, 'AllocationStatus'] = 'Unallocated'
                continue
            
            # Track total allocated for this transaction
            total_allocated_this_transaction = 0
            
            # Allocate across eligible grants until fully allocated or no grants left
            for grant_idx, grant in eligible_grants.iterrows():
                if remaining_amount <= 0:
                    break
                
                # Allocate as much as possible to this grant
                allocated_amount, remaining_amount = self._allocate_amount_to_grant(
                    grant, remaining_amount
                )
                
                if allocated_amount > 0:
                    # Record allocation
                    allocation_record = {
                        'AllocationId': str(uuid4()),
                        'TransactionId': transaction_id,
                        'GrantCode': grant['GrantCode'],
                        'GrantName': grant['GrantName'],
                        'AllocatedAmount': allocated_amount,
                        'AllocationTimestamp': timestamp,
                        'AllocationBatchId': batch_id,
                        'AllocationOrder': len(self.allocations) + 1
                    }
                    self.allocations.append(allocation_record)
                    total_allocated_this_transaction += allocated_amount
                    
                    # Update grant's remaining budget
                    grant_idx_in_df = self.grants_df[
                        self.grants_df['GrantCode'] == grant['GrantCode']
                    ].index[0]
                    self.grants_df.at[grant_idx_in_df, 'RemainingBudget'] -= allocated_amount
            
            # Store the allocated amount for this transaction
            self.transactions_df.at[idx, 'AllocatedAmount'] = total_allocated_this_transaction
            
            # Track unallocated portion if any
            if remaining_amount > 0:
                self.transactions_df.at[idx, 'AllocationStatus'] = 'Partial'
                self.unallocated.append({
                    'TransactionId': transaction_id,
                    'Amount': remaining_amount,
                    'Reason': 'Insufficient grant budgets',
                    'TransactionDate': transaction['TransactionDate'],
                    'OriginalAmount': transaction['Amount'],
                    'AllocatedAmount': total_allocated_this_transaction
                })
            else:
                self.transactions_df.at[idx, 'AllocationStatus'] = 'Fully Allocated'
        
        # Mark allocation as complete
        self.allocation_complete = True
        
        # Create DataFrames for outputs
        allocations_df = pd.DataFrame(self.allocations) if self.allocations else pd.DataFrame()
        unallocated_df = pd.DataFrame(self.unallocated) if self.unallocated else pd.DataFrame()
        
        # Remove duplicate unallocated entries (if any)
        if not unallocated_df.empty and 'TransactionId' in unallocated_df.columns:
            unallocated_df = unallocated_df.drop_duplicates(subset=['TransactionId'], keep='first')
        
        # Calculate burn rate report
        burn_rate_df = self._calculate_burn_rate()
        
        # Validation check
        total_expenses = self.transactions_df['Amount'].sum()
        total_allocated_check = self.transactions_df['AllocatedAmount'].sum()
        total_unallocated_check = unallocated_df['Amount'].sum() if not unallocated_df.empty else 0
        
        # The sum of allocated + unallocated should equal total expenses (within rounding)
        if abs((total_allocated_check + total_unallocated_check) - total_expenses) > 0.01:
            print(f"Warning: Allocation mismatch detected!")
            print(f"  Total expenses: {total_expenses:,.2f}")
            print(f"  Total allocated: {total_allocated_check:,.2f}")
            print(f"  Total unallocated: {total_unallocated_check:,.2f}")
            print(f"  Sum: {total_allocated_check + total_unallocated_check:,.2f}")
        
        return allocations_df, unallocated_df, burn_rate_df


    def _calculate_burn_rate(self) -> pd.DataFrame:
        """
        Calculate burn rate for each grant (allocated amount / total amount).
        """
        if not self.allocations:
            allocations_df = pd.DataFrame(columns=['GrantCode', 'AllocatedAmount'])
        else:
            allocations_df = pd.DataFrame(self.allocations)

        # Aggregate allocations by grant
        allocated_summary = allocations_df.groupby('GrantCode')['AllocatedAmount'].sum().reset_index()
        allocated_summary.rename(columns={'AllocatedAmount': 'AllocatedAmount'}, inplace=True)

        # Merge with grants data
        burn_rate = self.grants_df[['GrantCode', 'GrantName', 'Priority', 'TotalAmount', 'RemainingBudget']].copy()
        burn_rate = burn_rate.merge(allocated_summary, on='GrantCode', how='left')
        burn_rate['AllocatedAmount'] = burn_rate['AllocatedAmount'].fillna(0)
        burn_rate['BurnRate'] = burn_rate['AllocatedAmount'] / burn_rate['TotalAmount']
        burn_rate['Utilization'] = (burn_rate['TotalAmount'] - burn_rate['RemainingBudget']) / burn_rate['TotalAmount']
        burn_rate['Status'] = burn_rate.apply(
            lambda row: 'Exhausted' if row['RemainingBudget'] <= 0 else 'Active',
            axis=1
        )

        return burn_rate

    def _generate_batch_id(self) -> str:
        """
        Generate a unique batch ID for idempotency tracking.
        """
        # Create a hash of the input data for deterministic batch ID
        try:
            grants_hash = hashlib.md5(
                pd.util.hash_pandas_object(self.grants_df, index=True).values.tobytes()
            ).hexdigest()[:8]

            transactions_hash = hashlib.md5(
                pd.util.hash_pandas_object(self.transactions_df, index=True).values.tobytes()
            ).hexdigest()[:8]
        except:
            # Fallback for older pandas versions
            grants_hash = hashlib.md5(str(self.grants_df.shape).encode()).hexdigest()[:8]
            transactions_hash = hashlib.md5(str(self.transactions_df.shape).encode()).hexdigest()[:8]

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        return f"batch_{timestamp}_{grants_hash}_{transactions_hash}"

    def generate_summary_report(self) -> Dict:
        """
        Generate a comprehensive summary report of the allocation results.
        """
        allocations_df, unallocated_df, burn_rate_df = self.run_allocation()

        total_expenses = self.transactions_df['Amount'].sum()
        total_allocated = allocations_df['AllocatedAmount'].sum() if not allocations_df.empty else 0
        total_unallocated = unallocated_df['Amount'].sum() if not unallocated_df.empty else 0

        report = {
            'summary': {
                'total_expenses': float(total_expenses),
                'total_allocated': float(total_allocated),
                'total_unallocated': float(total_unallocated),
                'allocation_rate': (total_allocated / total_expenses * 100) if total_expenses > 0 else 0,
                'unallocated_rate': (total_unallocated / total_expenses * 100) if total_expenses > 0 else 0,
                'transactions_processed': len(self.transactions_df),
                'transactions_fully_allocated': len(self.transactions_df[self.transactions_df['AllocationStatus'] == 'Fully Allocated']),
                'transactions_partially_allocated': len(self.transactions_df[self.transactions_df['AllocationStatus'] == 'Partial']),
                'transactions_unallocated': len(unallocated_df),
                'number_of_allocations': len(self.allocations),
                'grants_utilized': len(burn_rate_df[burn_rate_df['AllocatedAmount'] > 0]),
                'grants_exhausted': len(burn_rate_df[burn_rate_df['Status'] == 'Exhausted'])
            },
            'burn_rate_summary': burn_rate_df.to_dict('records') if not burn_rate_df.empty else [],
            'unallocated_transactions': unallocated_df.to_dict('records') if not unallocated_df.empty else [],
            'allocations_preview': allocations_df.head(10).to_dict('records') if not allocations_df.empty else []
        }

        return report

    def save_outputs(self, output_dir: str = 'allocation_outputs'):
        """
        Save all outputs to CSV files for audit and reporting.
        """
        os.makedirs(output_dir, exist_ok=True)

        allocations_df, unallocated_df, burn_rate_df = self.run_allocation()

        # Save main outputs
        if not allocations_df.empty:
            allocations_df.to_csv(f'{output_dir}/allocations.csv', index=False)

        if not unallocated_df.empty:
            unallocated_df.to_csv(f'{output_dir}/unallocated_transactions.csv', index=False)

        if not burn_rate_df.empty:
            burn_rate_df.to_csv(f'{output_dir}/burn_rate_report.csv', index=False)

        # Save updated transactions with allocation status
        self.transactions_df.to_csv(f'{output_dir}/transactions_with_status.csv', index=False)

        # Save grants with remaining budgets
        self.grants_df.to_csv(f'{output_dir}/grants_remaining_budgets.csv', index=False)

        print(f" Outputs saved to '{output_dir}/' directory")


# Load data and run allocation engine.

def load_data_for_enviroment(grants_path: str, expenses_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load grants and expenses data for Enviroment.

    Args:
        grants_path: Path to grants CSV file
        expenses_path: Path to expenses CSV file

    Returns:
        Tuple of (grants_df, transactions_df)
    """
    print(f"\n Attempting to load files...")
    print(f"   Grants file: {grants_path}")
    print(f"   Expenses file: {expenses_path}")

    # Check if files exist
    if not os.path.exists(grants_path):
        raise FileNotFoundError(f"Grants file not found at: {grants_path}")

    if not os.path.exists(expenses_path):
        raise FileNotFoundError(f"Expenses file not found at: {expenses_path}")

    # Load CSV files
    grants_df = pd.read_csv(grants_path)
    expenses_df = pd.read_csv(expenses_path)

    # Clean up column names (remove any whitespace)
    grants_df.columns = grants_df.columns.str.strip()
    expenses_df.columns = expenses_df.columns.str.strip()

    # Handle missing values - replace NaN with None for proper wildcard handling
    grants_df = grants_df.replace({np.nan: None, '': None})
    expenses_df = expenses_df.replace({np.nan: None, '': None})

    # Convert date columns if they exist
    date_columns = ['StartDate', 'EndDate']
    for col in date_columns:
        if col in grants_df.columns:
            grants_df[col] = pd.to_datetime(grants_df[col])

    if 'TransactionDate' in expenses_df.columns:
        expenses_df['TransactionDate'] = pd.to_datetime(expenses_df['TransactionDate'])

    return grants_df, expenses_df


def load_data_run_engine(grants_file: str = 'grants.csv',
                  expenses_file: str = 'expenses.csv',
                  output_dir: str = 'allocation_outputs'):
    """
    Main function to run the allocation engine in Enviroment.

    Args:
        grants_file: Name of grants CSV file in the current directory
        expenses_file: Name of expenses CSV file in the current directory
        output_dir: Directory to save output files
    """

    print("=" * 80)
    print("GRANT EXPENSE ALLOCATION ENGINE")
    print("One Acre Fund - Data Engineering Exercise")
    print("Enviroment Version")
    print("=" * 80)

    # Get the current working directory (where the script is located)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"\n Script directory: {script_dir}")

    # List CSV files in the script directory
    print(f"\  CSV files in script directory:")
    csv_files = [f for f in os.listdir(script_dir) if f.endswith('.csv')]
    if csv_files:
        for file in csv_files:
            print(f"   - {file}")
    else:
        print("   No CSV files found!")

    # Construct full paths
    grants_path = os.path.join(script_dir, grants_file)
    expenses_path = os.path.join(script_dir, expenses_file)

    try:
        # Load data
        grants_df, expenses_df = load_data_for_enviroment(grants_path, expenses_path)

        print(f"\n Successfully loaded:")
        print(f"   - {len(grants_df)} grants")
        print(f"   - {len(expenses_df)} transactions")

        # Display column info for verification
        print(f"\n Grants columns: {list(grants_df.columns)}")
        print(f" Expenses columns: {list(expenses_df.columns)}")

        # Display sample of data
        if len(grants_df) > 0:
            print(f"\n Grants sample (first 3 rows):")
            print(grants_df.head(3).to_string())
        if len(expenses_df) > 0:
            print(f"\n Expenses sample (first 3 rows):")
            print(expenses_df.head(3).to_string())

    except FileNotFoundError as e:
        print(f"\n Error: {e}")
        print("\n Troubleshooting tips:")
        print("   1. Make sure 'grants.csv' and 'expenses.csv' are in the same folder as this script")
        print(f"   2. Current script folder: {script_dir}")
        print("   3. Check that the filenames match exactly (case-sensitive)")
        return None, None
    except Exception as e:
        print(f"\n Error loading data: {e}")
        import traceback
        traceback.print_exc()
        return None, None

    # Initialize and run engine
    print("\n⚙️ Running allocation engine...")
    engine = GrantAllocationEngine(grants_df, expenses_df)

    # Generate report
    report = engine.generate_summary_report()

    # Display summary
    print("\n" + "=" * 80)
    print("ALLOCATION SUMMARY REPORT")
    print("=" * 80)

    summary = report['summary']
    print(f"\n Overall Statistics:")
    print(f"   Total Expenses:        ${summary['total_expenses']:,.2f}")
    print(f"   Total Allocated:       ${summary['total_allocated']:,.2f}")
    print(f"   Total Unallocated:     ${summary['total_unallocated']:,.2f}")
    print(f"   Allocation Rate:       {summary['allocation_rate']:.1f}%")
    print(f"   Unallocated Rate:      {summary['unallocated_rate']:.1f}%")

    print(f"\n Transaction Status:")
    print(f"   Processed:             {summary['transactions_processed']}")
    print(f"   Fully Allocated:       {summary['transactions_fully_allocated']}")
    print(f"   Partially Allocated:   {summary['transactions_partially_allocated']}")
    print(f"   Unallocated:           {summary['transactions_unallocated']}")

    print(f"\n Grant Utilization:")
    print(f"   Number of Allocations: {summary['number_of_allocations']}")
    print(f"   Grants Utilized:       {summary['grants_utilized']}")
    print(f"   Grants Exhausted:      {summary['grants_exhausted']}")

    # Check for high unallocated balance and alert
    if summary['unallocated_rate'] > 5:
        print(f"\n⚠️  ALERT: High unallocated balance detected!")
        print(f"   Unallocated rate: {summary['unallocated_rate']:.1f}% exceeds 5% threshold")
        print(f"   Total unallocated amount: ${summary['total_unallocated']:,.2f}")
        print(f"   Please review unallocated_transactions.csv for details")

    # Save outputs
    output_path = os.path.join(script_dir, output_dir)
    print(f"\n Saving outputs to '{output_path}/' directory...")
    
    # Temporarily change save_outputs method to use absolute path
    original_save = engine.save_outputs
    def save_with_abs_path(*args, **kwargs):
        os.makedirs(output_path, exist_ok=True)
        allocations_df, unallocated_df, burn_rate_df = engine.run_allocation()
        if not allocations_df.empty:
            allocations_df.to_csv(f'{output_path}/allocations.csv', index=False)
        if not unallocated_df.empty:
            unallocated_df.to_csv(f'{output_path}/unallocated_transactions.csv', index=False)
        if not burn_rate_df.empty:
            burn_rate_df.to_csv(f'{output_path}/burn_rate_report.csv', index=False)
        engine.transactions_df.to_csv(f'{output_path}/transactions_with_status.csv', index=False)
        engine.grants_df.to_csv(f'{output_path}/grants_remaining_budgets.csv', index=False)
        print(f" Outputs saved to '{output_path}/' directory")
    
    engine.save_outputs = save_with_abs_path.__get__(engine)
    engine.save_outputs()

    print("\n Allocation complete!")
    print(f"\n Output files saved to '{output_path}/' directory:")
    print("   - allocations.csv (all allocation records)")
    print("   - unallocated_transactions.csv (unallocated amounts)")
    print("   - burn_rate_report.csv (grant burn rates)")
    print("   - transactions_with_status.csv (updated transactions)")
    print("   - grants_remaining_budgets.csv (updated grant budgets)")

    # Display output files
    if os.path.exists(output_path):
        print(f"\n Output files in '{output_path}/':")
        for file in os.listdir(output_path):
            if file.endswith('.csv'):
                file_path = os.path.join(output_path, file)
                file_size = os.path.getsize(file_path)
                print(f"   - {file} ({file_size:,} bytes)")

    return engine, report


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    engine, report = load_data_run_engine('grants.csv', 'expenses.csv')