# Grant Allocation Engine

A logic-driven system to allocate expenses to grants based on business rules.

## Overview

This engine processes transaction expenses and allocates them to eligible grants based on:
- Date range eligibility
- Budget availability
- Business unit, country, account, project, and department restrictions
- Priority-based allocation ordering

## Quick Start

### Prerequisites
- Python 3.8+
- pandas
- numpy

### Installation

```bash
git clone git@github.com:MutuaNdunda/grant-allocation-engine.git
cd grant-allocation-engine
pip install -r requirements.txt